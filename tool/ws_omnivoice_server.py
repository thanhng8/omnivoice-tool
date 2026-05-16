"""
OmniVoice TTS Server (HTTP + WebSocket)
========================================
Mirrors the architecture of supertonic/tool/ws_tts_server.py but exposes the
full OmniVoice feature set: voice cloning, voice design, auto voice, 646
languages, all generation parameters, and non-verbal markers.

Usage:
    cd OmniVoice/tool
    python ws_omnivoice_server.py                    # auto GPU/CPU
    python ws_omnivoice_server.py --port 8765 --cpu  # force CPU
    python ws_omnivoice_server.py --no-asr           # skip Whisper (saves VRAM)

Endpoints:
    GET  /                        -> serve omnivoice_web.html
    GET  /api/info                -> server capabilities (langs, voices, attrs)
    GET  /api/voices              -> list saved voice profiles
    POST /api/voices              -> upload+save voice profile (multipart form:
                                       fields: name, ref_text; file: audio)
    DELETE /api/voices/<name>     -> delete a profile
    GET  /api/voices/<name>/audio -> download the original ref audio
    WS   /ws                      -> TTS WebSocket
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import sys
import time
import re
import unicodedata
from pathlib import Path
from typing import Any, Optional

# ----------------------------------------------------------------------------
# Standard logging
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("omnivoice-server")

# ----------------------------------------------------------------------------
# Imports for OmniVoice / aiohttp
# ----------------------------------------------------------------------------
import numpy as np
import soundfile as sf
import torch
from aiohttp import web, WSMsgType

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, LANG_NAME_TO_ID, lang_display_name


# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
TOOL_DIR = Path(__file__).resolve().parent
VOICE_DIR = TOOL_DIR / "voice_prompts"
WEB_FILE = TOOL_DIR / "omnivoice_web.html"


# ----------------------------------------------------------------------------
# Voice Design attribute catalogue (mirrors omnivoice/cli/demo.py + voice-design.md)
# ----------------------------------------------------------------------------
VOICE_DESIGN = {
    "gender": {
        "label": "Gender / 性别",
        "options": [("male", "男"), ("female", "女")],
    },
    "age": {
        "label": "Age / 年龄",
        "options": [
            ("child", "儿童"),
            ("teenager", "少年"),
            ("young adult", "青年"),
            ("middle-aged", "中年"),
            ("elderly", "老年"),
        ],
    },
    "pitch": {
        "label": "Pitch / 音调",
        "options": [
            ("very low pitch", "极低音调"),
            ("low pitch", "低音调"),
            ("moderate pitch", "中音调"),
            ("high pitch", "高音调"),
            ("very high pitch", "极高音调"),
        ],
    },
    "style": {
        "label": "Style / 风格",
        "options": [("whisper", "耳语")],
    },
    "english_accent": {
        "label": "English Accent / 英文口音",
        "note": "Only effective when synthesis text is English.",
        "options": [
            ("american accent", "美式口音"),
            ("british accent", "英国口音"),
            ("australian accent", "澳大利亚口音"),
            ("canadian accent", "加拿大口音"),
            ("indian accent", "印度口音"),
            ("chinese accent", "中国口音"),
            ("korean accent", "韩国口音"),
            ("japanese accent", "日本口音"),
            ("portuguese accent", "葡萄牙口音"),
            ("russian accent", "俄罗斯口音"),
        ],
    },
    "chinese_dialect": {
        "label": "Chinese Dialect / 中文方言",
        "note": "Only effective when synthesis text is Chinese.",
        "options": [
            ("河南话", "河南话"),
            ("陕西话", "陕西话"),
            ("四川话", "四川话"),
            ("贵州话", "贵州话"),
            ("云南话", "云南话"),
            ("桂林话", "桂林话"),
            ("济南话", "济南话"),
            ("石家庄话", "石家庄话"),
            ("甘肃话", "甘肃话"),
            ("宁夏话", "宁夏话"),
            ("青岛话", "青岛话"),
            ("东北话", "东北话"),
        ],
    },
}

NONVERBAL_MARKERS = [
    "[laughter]", "[sigh]",
    "[confirmation-en]", "[question-en]",
    "[question-ah]", "[question-oh]", "[question-ei]", "[question-yi]",
    "[surprise-ah]", "[surprise-oh]", "[surprise-wa]", "[surprise-yo]",
    "[dissatisfaction-hnn]",
]

# Generation parameter defaults + bounds (for UI sliders / validation)
GEN_PARAM_SPECS = {
    "num_step":             {"default": 32,   "min": 4,   "max": 64,  "step": 1},
    "guidance_scale":       {"default": 2.0,  "min": 0.0, "max": 4.0, "step": 0.1},
    "t_shift":              {"default": 0.1,  "min": 0.0, "max": 1.0, "step": 0.05},
    "layer_penalty_factor": {"default": 5.0,  "min": 0.0, "max": 20.0,"step": 0.5},
    "position_temperature": {"default": 5.0,  "min": 0.0, "max": 20.0,"step": 0.5},
    "class_temperature":    {"default": 0.0,  "min": 0.0, "max": 2.0, "step": 0.05},
    "speed":                {"default": 1.0,  "min": 0.5, "max": 1.5, "step": 0.05},
    "duration":             {"default": None, "min": 0.5, "max": 60.0,"step": 0.5},
    "audio_chunk_duration": {"default": 15.0, "min": 5.0, "max": 30.0,"step": 1},
    "audio_chunk_threshold":{"default": 30.0, "min": 5.0, "max": 60.0,"step": 1},
    "denoise":              {"default": True},
    "preprocess_prompt":    {"default": True},
    "postprocess_output":   {"default": True},
}


# ----------------------------------------------------------------------------
# Voice profile manager
# ----------------------------------------------------------------------------
def _safe_name(name: str) -> str:
    """Return a filesystem-safe slug for a profile name."""
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^A-Za-z0-9_\- ]+", "", name).strip().replace(" ", "_")
    return name[:64] or "voice"


class VoiceProfileStore:
    """Load/save voice profiles from `voice_prompts/`.

    A profile lives as a pair of files:
        <name>.wav    - reference audio (any sr; OmniVoice resamples)
        <name>.json   - {"ref_text": "...", "language": "...", "note": "..."}

    Optionally, a sibling ``<name>.txt`` is read as ref_text for compatibility
    with the supertonic-style folder layout.
    """

    def __init__(self, model: OmniVoice, voice_dir: Path):
        self.model = model
        self.dir = voice_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.profiles: dict[str, dict] = {}     # name -> {"ref_text", "audio", "language"}
        self.prompts: dict[str, Any] = {}       # name -> VoiceClonePrompt (lazy)

    def scan(self):
        """(Re-)scan the voice_prompts directory."""
        self.profiles.clear()
        self.prompts.clear()
        for wav in sorted(self.dir.glob("*.wav")):
            name = wav.stem
            meta_path = wav.with_suffix(".json")
            txt_path = wav.with_suffix(".txt")
            meta = {}
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception as e:
                    log.warning("Bad JSON for %s: %s", name, e)
            ref_text = meta.get("ref_text")
            if not ref_text and txt_path.is_file():
                ref_text = txt_path.read_text(encoding="utf-8").strip()
            self.profiles[name] = {
                "name": name,
                "ref_text": ref_text or "",
                "language": meta.get("language"),
                "note": meta.get("note", ""),
                "audio_path": str(wav),
            }
        log.info("Loaded %d voice profile(s) from %s", len(self.profiles), self.dir)

    def list_meta(self) -> list[dict]:
        return [
            {k: v for k, v in p.items() if k != "audio_path"}
            for p in self.profiles.values()
        ]

    def get_audio_path(self, name: str) -> Optional[Path]:
        p = self.profiles.get(name)
        return Path(p["audio_path"]) if p else None

    def get_prompt(self, name: str, preprocess_prompt: bool = True):
        """Return a (cached) VoiceClonePrompt for the named profile."""
        cache_key = (name, preprocess_prompt)
        if cache_key in self.prompts:
            return self.prompts[cache_key]
        prof = self.profiles.get(name)
        if not prof:
            raise KeyError(f"Voice profile not found: {name}")
        prompt = self.model.create_voice_clone_prompt(
            ref_audio=prof["audio_path"],
            ref_text=prof["ref_text"] or None,
            preprocess_prompt=preprocess_prompt,
        )
        self.prompts[cache_key] = prompt
        return prompt

    def save(self, name: str, audio_bytes: bytes, ref_text: str = "",
             language: Optional[str] = None, note: str = "") -> dict:
        name = _safe_name(name)
        if not name:
            raise ValueError("Invalid profile name")
        wav_path = self.dir / f"{name}.wav"
        meta_path = self.dir / f"{name}.json"
        wav_path.write_bytes(audio_bytes)
        meta_path.write_text(
            json.dumps(
                {"ref_text": ref_text, "language": language, "note": note},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        # invalidate prompt cache for this name
        for k in list(self.prompts.keys()):
            if k[0] == name:
                self.prompts.pop(k, None)
        self.scan()
        return self.profiles[name]

    def delete(self, name: str) -> bool:
        prof = self.profiles.get(name)
        if not prof:
            return False
        for ext in (".wav", ".json", ".txt"):
            p = self.dir / f"{name}{ext}"
            if p.exists():
                p.unlink()
        self.profiles.pop(name, None)
        for k in list(self.prompts.keys()):
            if k[0] == name:
                self.prompts.pop(k, None)
        return True


# ----------------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------------
class TTSEngine:
    """Wraps OmniVoice with a serialised generate() (one inference at a time)."""

    def __init__(self, model_path: str, device: str, dtype: torch.dtype,
                 load_asr: bool, asr_model: str):
        log.info("Loading OmniVoice from %s on %s ...", model_path, device)
        t0 = time.time()
        self.model = OmniVoice.from_pretrained(
            model_path,
            device_map=device,
            dtype=dtype,
            load_asr=load_asr,
            asr_model_name=asr_model,
        )
        log.info("Loaded in %.1fs", time.time() - t0)
        self.sampling_rate = int(self.model.sampling_rate)
        self.device = device
        self._lock = asyncio.Lock()

        # Voice profile store
        self.voices = VoiceProfileStore(self.model, VOICE_DIR)
        self.voices.scan()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_language(lang: Optional[str]) -> Optional[str]:
        """Accept display name (e.g. 'English'), lowercase ('english'),
        OmniVoice ID ('en') or ISO 639-3 ('eng'). Return what generate() expects.
        Returns None for empty/Auto."""
        if not lang or lang.lower() in {"auto", "none", ""}:
            return None
        l = lang.strip()
        # Already an OmniVoice ID or full name? Pass through; generate() handles it.
        return l

    def _build_instruct(self, parts: list[str]) -> Optional[str]:
        parts = [p.strip() for p in parts if p and p.strip()]
        return ", ".join(parts) if parts else None

    # ------------------------------------------------------------------
    # Synthesise one item — runs in a worker thread
    # ------------------------------------------------------------------
    def _synthesise_blocking(self, req: dict) -> tuple[bytes, dict]:
        text = req.get("text", "").strip()
        if not text:
            raise ValueError("Empty text")

        # Resolve language
        language = self._resolve_language(req.get("lang"))

        # Build generation_config from any provided fields
        cfg_kwargs = {}
        for key in (
            "num_step", "guidance_scale", "t_shift",
            "layer_penalty_factor", "position_temperature", "class_temperature",
            "denoise", "preprocess_prompt", "postprocess_output",
            "audio_chunk_duration", "audio_chunk_threshold",
        ):
            if key in req and req[key] is not None:
                cfg_kwargs[key] = req[key]
        gen_cfg = OmniVoiceGenerationConfig.from_dict(cfg_kwargs)

        # Mode dispatch
        mode = (req.get("mode") or "auto").lower()
        kw: dict[str, Any] = dict(
            text=text,
            language=language,
            generation_config=gen_cfg,
        )

        # speed / duration
        if req.get("speed") is not None and float(req["speed"]) != 1.0:
            kw["speed"] = float(req["speed"])
        if req.get("duration") is not None and float(req["duration"]) > 0:
            kw["duration"] = float(req["duration"])

        # Voice clone via saved profile name
        if mode == "clone":
            voice_name = req.get("voice")
            if voice_name:
                prompt = self.voices.get_prompt(
                    voice_name,
                    preprocess_prompt=gen_cfg.preprocess_prompt,
                )
                kw["voice_clone_prompt"] = prompt
            elif req.get("ref_audio_path"):
                kw["voice_clone_prompt"] = self.model.create_voice_clone_prompt(
                    ref_audio=req["ref_audio_path"],
                    ref_text=req.get("ref_text") or None,
                    preprocess_prompt=gen_cfg.preprocess_prompt,
                )
            else:
                raise ValueError("Clone mode requires `voice` (saved profile) or `ref_audio_path`")

        # Voice design
        if mode == "design":
            instruct = req.get("instruct") or self._build_instruct(
                req.get("instruct_parts") or []
            )
            if instruct:
                kw["instruct"] = instruct

        # Auto mode = neither clone nor design extras

        # Inference
        t0 = time.perf_counter()
        with torch.inference_mode():
            audios = self.model.generate(**kw)
        latency = time.perf_counter() - t0

        wav = audios[0]
        # Encode to WAV bytes (PCM 16-bit)
        buf = io.BytesIO()
        sf.write(buf, wav, self.sampling_rate, format="WAV", subtype="PCM_16")
        meta = {
            "duration": round(len(wav) / self.sampling_rate, 3),
            "latency_ms": round(latency * 1000),
            "sample_rate": self.sampling_rate,
            "size": buf.tell(),
            "mode": mode,
            "language": language,
        }
        return buf.getvalue(), meta

    async def synthesise(self, req: dict) -> tuple[bytes, dict]:
        """Async wrapper: lock GPU + run blocking inference in a worker."""
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._synthesise_blocking, req)


# ----------------------------------------------------------------------------
# HTTP / WebSocket app
# ----------------------------------------------------------------------------
def make_info_payload(engine: TTSEngine) -> dict:
    """Server capabilities sent to clients (handshake + /api/info)."""
    languages = [
        {"name": lang_display_name(n), "id": LANG_NAME_TO_ID[n]}
        for n in sorted(LANG_NAMES)
    ]
    voice_design = {
        k: {
            "label": v["label"],
            "note": v.get("note"),
            "options": v["options"],   # list of (en, zh)
        }
        for k, v in VOICE_DESIGN.items()
    }
    return {
        "status": "connected",
        "device": engine.device,
        "sample_rate": engine.sampling_rate,
        "languages": languages,                           # 646 langs
        "voices": engine.voices.list_meta(),              # saved profiles
        "voice_design": voice_design,
        "nonverbal_markers": NONVERBAL_MARKERS,
        "gen_params": GEN_PARAM_SPECS,
        "modes": ["auto", "clone", "design"],
    }


# --------------------- HTTP handlers ----------------------------------------
async def index(request: web.Request) -> web.Response:
    if WEB_FILE.is_file():
        return web.FileResponse(WEB_FILE, headers={"Cache-Control": "no-store"})
    return web.Response(text="omnivoice_web.html missing", status=404)


async def api_info(request: web.Request) -> web.Response:
    engine: TTSEngine = request.app["engine"]
    return web.json_response(make_info_payload(engine))


async def api_list_voices(request: web.Request) -> web.Response:
    engine: TTSEngine = request.app["engine"]
    return web.json_response({"voices": engine.voices.list_meta()})


async def api_upload_voice(request: web.Request) -> web.Response:
    engine: TTSEngine = request.app["engine"]
    reader = await request.multipart()
    name = ""
    ref_text = ""
    language = None
    note = ""
    audio_bytes = b""
    audio_filename = ""

    while True:
        field = await reader.next()
        if field is None:
            break
        if field.name == "name":
            name = (await field.text()).strip()
        elif field.name == "ref_text":
            ref_text = (await field.text()).strip()
        elif field.name == "language":
            language = (await field.text()).strip() or None
        elif field.name == "note":
            note = (await field.text()).strip()
        elif field.name == "audio":
            audio_filename = field.filename or "voice.wav"
            chunks = []
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                chunks.append(chunk)
            audio_bytes = b"".join(chunks)

    if not name or not audio_bytes:
        return web.json_response(
            {"error": "Missing 'name' or 'audio'"}, status=400
        )
    try:
        prof = engine.voices.save(
            name, audio_bytes, ref_text=ref_text,
            language=language, note=note,
        )
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)
    return web.json_response({
        "ok": True,
        "voice": {k: v for k, v in prof.items() if k != "audio_path"},
        "filename": audio_filename,
    })


async def api_delete_voice(request: web.Request) -> web.Response:
    engine: TTSEngine = request.app["engine"]
    name = request.match_info["name"]
    ok = engine.voices.delete(_safe_name(name))
    return web.json_response({"ok": ok})


async def api_voice_audio(request: web.Request) -> web.Response:
    engine: TTSEngine = request.app["engine"]
    name = _safe_name(request.match_info["name"])
    p = engine.voices.get_audio_path(name)
    if not p or not p.is_file():
        return web.Response(text="not found", status=404)
    return web.FileResponse(p, headers={"Content-Type": "audio/wav"})


# --------------------- WebSocket handler ------------------------------------
async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    engine: TTSEngine = request.app["engine"]
    ws = web.WebSocketResponse(max_msg_size=20 * 1024 * 1024, heartbeat=20)
    await ws.prepare(request)
    log.info("[WS] client connected from %s", request.remote)

    # Send capabilities handshake
    try:
        await ws.send_str(json.dumps(make_info_payload(engine)))
    except Exception:
        return ws

    pending: list[asyncio.Task] = []

    async def process(req: dict, request_id):
        try:
            wav_bytes, meta = await engine.synthesise(req)
            if ws.closed:
                return
            meta_msg = {
                "type": "audio_meta",
                "request_id": request_id,
                "text": req.get("text", "")[:200],
                **meta,
            }
            await ws.send_str(json.dumps(meta_msg))
            await ws.send_bytes(wav_bytes)
        except Exception as e:
            log.exception("Synthesis failed")
            if not ws.closed:
                try:
                    await ws.send_str(json.dumps({
                        "type": "error",
                        "request_id": request_id,
                        "message": f"{type(e).__name__}: {e}",
                    }))
                except Exception:
                    pass

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    req = json.loads(msg.data)
                except json.JSONDecodeError as e:
                    await ws.send_str(json.dumps({"type": "error", "message": f"Bad JSON: {e}"}))
                    continue
                request_id = req.get("request_id")
                if not (req.get("text") or "").strip():
                    continue
                # Process concurrently (engine.synthesise has its own GPU lock)
                pending.append(asyncio.create_task(process(req, request_id)))
            elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED, WSMsgType.ERROR):
                break
    finally:
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    log.info("[WS] client disconnected")
    return ws


# ----------------------------------------------------------------------------
# CORS middleware (loopback dev only)
# ----------------------------------------------------------------------------
@web.middleware
async def cors_middleware(request: web.Request, handler):
    if request.method == "OPTIONS":
        resp = web.Response(status=204)
    else:
        resp = await handler(request)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


def make_app(engine: TTSEngine) -> web.Application:
    app = web.Application(middlewares=[cors_middleware], client_max_size=50 * 1024 * 1024)
    app["engine"] = engine
    app.router.add_get("/", index)
    app.router.add_get("/api/info", api_info)
    app.router.add_get("/api/voices", api_list_voices)
    app.router.add_post("/api/voices", api_upload_voice)
    app.router.add_delete("/api/voices/{name}", api_delete_voice)
    app.router.add_get("/api/voices/{name}/audio", api_voice_audio)
    app.router.add_get("/ws", ws_handler)
    return app


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def get_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser(description="OmniVoice WebSocket TTS server")
    p.add_argument("--model", default="k2-fsa/OmniVoice")
    p.add_argument("--ip", default="127.0.0.1")
    p.add_argument("--port", type=int, default=int(os.environ.get("OMV_PORT", 8765)))
    p.add_argument("--device", default=None,
                   help="cuda | mps | cpu (auto-detected if omitted)")
    p.add_argument("--cpu", action="store_true", help="Force CPU.")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "float32", "bfloat16"])
    p.add_argument("--no-asr", action="store_true",
                   help="Skip Whisper ASR (saves VRAM; ref_text becomes mandatory).")
    p.add_argument("--asr-model", default="openai/whisper-large-v3-turbo")
    args = p.parse_args()

    device = "cpu" if args.cpu else (args.device or get_best_device())
    dtype = {"float16": torch.float16, "float32": torch.float32,
             "bfloat16": torch.bfloat16}[args.dtype]
    if device == "cpu" and dtype == torch.float16:
        log.info("CPU + float16 not stable; falling back to float32.")
        dtype = torch.float32

    engine = TTSEngine(
        model_path=args.model,
        device=device,
        dtype=dtype,
        load_asr=not args.no_asr,
        asr_model=args.asr_model,
    )

    app = make_app(engine)

    print("=" * 60)
    print(" OmniVoice TTS Server")
    print(f"   HTTP/WS:  http://{args.ip}:{args.port}/")
    print(f"   WebSocket: ws://{args.ip}:{args.port}/ws")
    print(f"   Device:   {engine.device}, dtype={args.dtype}, sr={engine.sampling_rate} Hz")
    print(f"   Voices:   {len(engine.voices.profiles)} profile(s)"
          f" in {VOICE_DIR}")
    print(f"   Languages: 646 supported (see /api/info)")
    print("=" * 60)
    web.run_app(app, host=args.ip, port=args.port, print=lambda *_: None,
                access_log=None)


if __name__ == "__main__":
    main()
