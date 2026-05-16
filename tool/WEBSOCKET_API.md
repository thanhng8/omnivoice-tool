# OmniVoice TTS — HTTP + WebSocket API

Local server that exposes the full **OmniVoice** feature set
(zero-shot TTS for **646 languages**, voice cloning, voice design,
non-verbal markers, all generation parameters) over a single port for
native apps, web apps, and browser extensions.

This is a port of `supertonic/tool/ws_tts_server.py` adapted for the
much richer OmniVoice model.

---

## 1. Architecture at a glance

```
┌──────────────────────────────────────────────────────────┐
│  http://127.0.0.1:8765                                   │
│                                                          │
│   GET  /                  → omnivoice_web.html (UI)      │
│   GET  /api/info          → capabilities (langs, voices) │
│   GET  /api/voices        → list saved voice profiles    │
│   POST /api/voices        → upload + save a profile      │
│   DEL  /api/voices/<n>    → delete                       │
│   GET  /api/voices/<n>/audio → download original WAV     │
│                                                          │
│   WS   /ws                → TTS WebSocket                │
└──────────────────────────────────────────────────────────┘
```

| Property        | Value                                  |
| --------------- | -------------------------------------- |
| URL             | `http://127.0.0.1:8765`                |
| WebSocket URL   | `ws://127.0.0.1:8765/ws`               |
| Audio format    | WAV, 16-bit PCM mono, 24 kHz           |
| Max WS frame    | 20 MB                                  |
| Heartbeat       | 20 s (server → client)                 |
| Concurrency     | One inference at a time (GPU lock); requests queue |

The server binds to `127.0.0.1` only.

---

## 2. Three generation modes

| Mode | Required fields | Description |
|---|---|---|
| `auto` | `text` | Model picks a voice on its own. Cheapest mode. |
| `clone` | `text` + (`voice` profile name **or** `ref_audio_path`) | Zero-shot voice cloning from saved profile or local WAV path. |
| `design` | `text` + `instruct` | Describe gender/age/pitch/accent/dialect — no reference audio needed. |

You can mix modes per request — each WebSocket message is independent.

---

## 3. WebSocket: client → server

Send a single JSON frame per request. The server processes them serially
behind a GPU lock and replies in the same order with `audio_meta` + binary.

```jsonc
{
  "request_id":   42,                  // optional, echoed back; helpful for matching
  "text":         "Hello world",       // REQUIRED
  "lang":         "English",           // language name, ID, or null/"" for Auto
  "mode":         "clone",             // auto | clone | design (default "auto")

  // Voice clone
  "voice":          "alice",           // saved profile name (see /api/voices)
  "ref_audio_path": null,              // OR absolute path on the server's machine
  "ref_text":       null,              // optional transcript, paired with ref_audio_path

  // Voice design
  "instruct": "female, young adult, high pitch, british accent",

  // Generation params (all optional — server defaults match docs/generation-parameters.md)
  "num_step":              32,         // 4..64
  "guidance_scale":        2.0,        // 0..4
  "t_shift":               0.1,
  "layer_penalty_factor":  5.0,
  "position_temperature":  5.0,
  "class_temperature":     0.0,
  "denoise":               true,
  "preprocess_prompt":     true,
  "postprocess_output":    true,
  "audio_chunk_duration":  15.0,
  "audio_chunk_threshold": 30.0,

  // Duration / speed (duration > speed if both set)
  "speed":    1.0,                     // 0.5..1.5
  "duration": null                     // seconds; null lets the model estimate
}
```

### Field details

| Field | Type | Notes |
|---|---|---|
| `text` | string | **Required**. Inline markers supported — see §6 |
| `lang` | string \| null | Display name (`"Vietnamese"`), OmniVoice ID (`"vi"`), or ISO 639-3 (`"vie"`). `null` / `"Auto"` triggers auto-detection. |
| `mode` | string | `"auto"` (default), `"clone"`, or `"design"` |
| `voice` | string | Name of a profile saved in `tool/voice_prompts/`. Required for `clone` mode unless `ref_audio_path` is given. |
| `ref_audio_path` | string | Absolute path on the **server** filesystem; alternative to `voice` for one-shot cloning. |
| `ref_text` | string | Transcript of `ref_audio_path`. Optional — omit to auto-transcribe with Whisper (if `--no-asr` was not set at startup). |
| `instruct` | string | Comma-separated speaker attributes for `design` mode. See §5. |
| `num_step` … | various | See [`docs/generation-parameters.md`](../docs/generation-parameters.md) for full semantics. |

---

## 4. WebSocket: server → client

### 4.1 Handshake (sent once on connect)

```json
{
  "status": "connected",
  "device": "cuda",
  "sample_rate": 24000,
  "modes": ["auto", "clone", "design"],
  "languages": [{"name": "English", "id": "en"}, ...],          // 646 entries
  "voices":    [{"name": "alice", "ref_text": "...", ...}],     // saved profiles
  "voice_design": { "gender": {...}, "age": {...}, ... },       // attribute catalogue
  "nonverbal_markers": ["[laughter]", "[sigh]", ...],
  "gen_params": { "num_step": {"default":32, "min":4, "max":64}, ... }
}
```

Use this to populate UI dropdowns, etc.

### 4.2 Audio metadata (one per request, before the binary frame)

```json
{
  "type": "audio_meta",
  "request_id": 42,
  "text": "Hello world",         // echoed (truncated to 200 chars)
  "duration": 1.872,             // seconds of generated audio
  "latency_ms": 1842,            // server-side inference time
  "sample_rate": 24000,
  "size": 89868,                 // bytes of the next binary frame
  "mode": "clone",
  "language": "English"
}
```

### 4.3 Audio payload (binary)

The frame **immediately following** an `audio_meta` is a complete WAV file
(header + PCM 16-bit data, mono, 24 kHz). Treat as `Blob` / `ArrayBuffer`.

### 4.4 Error

```json
{ "type": "error", "request_id": 42, "message": "ValueError: Empty text" }
```

The connection stays open after errors; clients can retry.

---

## 5. Voice Design attributes

Voice Design lets you describe a voice with one tag per category. The
exact strings the server sends to OmniVoice come from
[`docs/voice-design.md`](../docs/voice-design.md):

| Category | Examples |
|---|---|
| **Gender** | `male`, `female` |
| **Age** | `child`, `teenager`, `young adult`, `middle-aged`, `elderly` |
| **Pitch** | `very low pitch`, `low pitch`, `moderate pitch`, `high pitch`, `very high pitch` |
| **Style** | `whisper` |
| **English Accent** *(English text only)* | `american accent`, `british accent`, `australian accent`, `canadian accent`, `indian accent`, `chinese accent`, `korean accent`, `japanese accent`, `portuguese accent`, `russian accent` |
| **Chinese Dialect** *(Chinese text only)* | `河南话`, `陕西话`, `四川话`, `贵州话`, `云南话`, `桂林话`, `济南话`, `石家庄话`, `甘肃话`, `宁夏话`, `青岛话`, `东北话` |

Combine across categories with commas:
```
"male, elderly, low pitch, whisper"
"female, young adult, 四川话"
```

---

## 6. Inline markers in `text`

| Marker | Effect |
|---|---|
| `[laughter]`, `[sigh]` | non-verbal sounds |
| `[confirmation-en]`, `[question-en]` | confirmation / question intonations |
| `[question-ah]` `[question-oh]` `[question-ei]` `[question-yi]` | rising-tone questions |
| `[surprise-ah]` `[surprise-oh]` `[surprise-wa]` `[surprise-yo]` | surprise reactions |
| `[dissatisfaction-hnn]` | dissatisfaction grunt |
| Pinyin override (Chinese) | `打ZHE2出售` → uppercase pinyin + tone digit |
| CMU override (English) | `[B EY1 S]` (in brackets) → ARPAbet phonemes |

All markers are sent verbatim — no escaping needed.

---

## 7. Voice profiles (HTTP)

Profiles live in `tool/voice_prompts/<name>.wav` paired with a sibling
`<name>.json` (and optionally a legacy `<name>.txt` for ref_text). The
server scans this folder on startup and after every upload.

### 7.1 List

```http
GET /api/voices
→ { "voices": [{"name":"alice","ref_text":"...","language":"English","note":""}, ...] }
```

### 7.2 Upload (multipart form)

```http
POST /api/voices
Content-Type: multipart/form-data

  name=alice
  ref_text=Hello, this is a test reference recording.
  language=English          (optional)
  note=energetic young woman (optional)
  audio=@alice.wav          (required — WAV/MP3/FLAC/OGG/M4A)
```

Response: `{ "ok": true, "voice": {...} }`.

### 7.3 Delete

```http
DELETE /api/voices/alice
→ { "ok": true }
```

### 7.4 Preview audio

```http
GET /api/voices/alice/audio
→ original WAV bytes
```

---

## 8. Quick-reference examples

### 8.1 Vanilla JS (web)

```js
const ws = new WebSocket("ws://127.0.0.1:8765/ws");
ws.binaryType = "arraybuffer";

let server = {}, pendingMeta = null;
ws.onmessage = (ev) => {
  if (typeof ev.data === "string") {
    const m = JSON.parse(ev.data);
    if (m.status === "connected") server = m;
    else if (m.type === "audio_meta") pendingMeta = m;
    else if (m.type === "error") console.error(m.message);
  } else {
    const blob = new Blob([ev.data], {type: "audio/wav"});
    new Audio(URL.createObjectURL(blob)).play();
  }
};

function speak(text, opts = {}) {
  ws.send(JSON.stringify({ text, ...opts }));
}

// Voice design
speak("This is a test.", { mode: "design", instruct: "female, british accent" });

// Voice clone
speak("Hello, this is a Vietnamese voice clone test.", { mode: "clone", voice: "alice", lang: "Vietnamese" });
```

### 8.2 Python client

```python
import asyncio, json, websockets

async def synth(text, **opts):
    async with websockets.connect("ws://127.0.0.1:8765/ws") as ws:
        await ws.recv()                                       # discard handshake
        await ws.send(json.dumps({"text": text, **opts}))
        meta = json.loads(await ws.recv())
        wav = await ws.recv()                                 # bytes
        with open("out.wav", "wb") as f: f.write(wav)
        return meta

asyncio.run(synth(
    "Voice design test.",
    mode="design",
    instruct="male, british accent",
    num_step=16,
))
```

### 8.3 Chrome extension (MV3)

Same pattern as the supertonic example — connect from the background
service-worker, hand binary frames to an offscreen document for playback.
The handshake `voices` / `languages` arrays let you populate UI without
hard-coding.

---

## 9. Operational notes

### GPU memory
On 6 GB GPUs (e.g. GTX 1660 SUPER), keep `dtype=float16`, set
`--no-asr` to skip Whisper, and prefer `num_step=16` for faster runs.
The server uses `torch.inference_mode()` and one inference at a time to
avoid OOM.

### Concurrency
Multiple WS clients can connect; requests are processed FIFO behind a
single GPU lock. The web UI sends lines sequentially per the user's
"Convert all" button, but several clients can queue work concurrently.

### Cold start
Loading OmniVoice + Whisper takes 5–30 s the first time (HF download).
Subsequent loads are 2–5 s from the local HF cache.

### Security
- The server is **unauthenticated** and binds to loopback only.
- For exposing remotely, put a TLS-terminating reverse proxy in front
  (Caddy / nginx) and add auth.
- From `https://` pages, plain `ws://` is blocked by browsers; either
  serve the page over `http://localhost` (recommended) or use the proxy.

### Troubleshooting

| Symptom | Likely cause |
|---|---|
| Connection closes immediately | Server not running / port 8765 taken / firewall |
| `Clone mode requires 'voice' or 'ref_audio_path'` | You set `mode:"clone"` but didn't pass either field |
| Vietnamese sounds anglicised | Pass `lang: "Vietnamese"` explicitly; auto-detect can fail on short text |
| `RuntimeError: CUDA out of memory` | Lower `num_step`, drop `--no-asr`, or run with `--cpu` |
| Voice profile missing after upload | The audio extension isn't .wav/.mp3/.flac/.ogg/.m4a |
| Whisper auto-transcribe disabled | The server was started with `--no-asr`; pass `ref_text` explicitly |

---

## 10. Tiny cheat-sheet

```text
CONNECT     ws://127.0.0.1:8765/ws
RECV JSON   { status:"connected", languages, voices, voice_design, ... }

SEND JSON   { text, mode?, voice?|instruct?, lang?, ...gen_params }
   ...      (send as many as you like; server queues)

RECV JSON   { type:"audio_meta", request_id?, duration, latency_ms, ... }
RECV BIN    <WAV bytes, 16-bit PCM mono 24 kHz>

ON ERROR    RECV JSON { type:"error", request_id?, message }
```
