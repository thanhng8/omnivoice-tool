"""Pre-build a Voice Gallery for OmniVoice.

OmniVoice is zero-shot — there is NO fixed voice library. This script
pre-generates a curated set of voices and saves each as a cloneable profile
in ``voice_prompts/``, so users can browse them in the web UI like presets.

Strategy
--------
* **English / Chinese**: Voice Design (gender × age × pitch × accent/dialect)
  → produces deterministic, well-described voices.
* **Other languages (e.g. Vietnamese)**: two complementary packs —
  - **Random pack**: Auto Voice × N. Each result is automatically classified
    by median pitch (F0) into male/female and renamed
    ``<lang>_female_NN`` / ``<lang>_male_NN`` (sorted by pitch within group).
  - **Cross-lingual design pack** *(--xlingual)*: Voice Design with English
    text generates a controlled reference, then cross-lingual cloning to the
    target language produces the actual profile audio. This gives named
    voices (`female_young`, `male_elderly`, …) at the cost of a slight
    English accent.

Each profile is saved as:
    voice_prompts/<name>.wav      (the generated reference audio)
    voice_prompts/<name>.json     ({"ref_text", "language", "note", "tags"})

Usage
-----
    # Quick demo: 12 random Vietnamese voices, gender-labelled
    python build_voice_gallery.py --languages vi --random-voices 12

    # With cross-lingual design pack (named female/male voices, EN-accent)
    python build_voice_gallery.py --languages vi --random-voices 12 --xlingual

    # All popular languages
    python build_voice_gallery.py --languages vi,en,zh,ja,ko,fr,de,es,it,ru \\
        --random-voices 8 --xlingual
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
log = logging.getLogger("voice-gallery")

TOOL_DIR = Path(__file__).resolve().parent
VOICE_DIR = TOOL_DIR / "voice_prompts"
VOICE_DIR.mkdir(parents=True, exist_ok=True)


# Sample text per language used for generation. Kept short (~6–8 words) so the
# clip is suitable as a voice reference.
SAMPLE_TEXT = {
    "en":  "Hello, this is a sample voice for the gallery.",
    "zh":  "你好，这是声音库的一个样本声音。",
    "vi":  "Xin chào, đây là một giọng mẫu trong thư viện.",
    "ja":  "こんにちは、これはサンプルの声です。",
    "ko":  "안녕하세요, 이것은 샘플 보이스입니다.",
    "fr":  "Bonjour, voici un échantillon de voix.",
    "de":  "Hallo, dies ist eine Beispielstimme.",
    "es":  "Hola, esta es una muestra de voz.",
    "it":  "Ciao, questa è una voce di esempio.",
    "pt":  "Olá, esta é uma amostra de voz.",
    "ru":  "Здравствуйте, это образец голоса.",
    "id":  "Halo, ini adalah contoh suara.",
    "th":  "สวัสดี นี่เป็นตัวอย่างเสียง",
    "ar":  "مرحبا، هذا نموذج صوتي.",
    "hi":  "नमस्ते, यह एक नमूना आवाज़ है।",
    "tr":  "Merhaba, bu bir örnek ses.",
    "nl":  "Hallo, dit is een voorbeeldstem.",
    "pl":  "Cześć, to jest przykładowy głos.",
    "uk":  "Привіт, це зразок голосу.",
    "sv":  "Hej, detta är en exempelröst.",
}

LANG_NAME_FOR_GENERATE = {
    "en": "English", "zh": "Chinese", "vi": "Vietnamese", "ja": "Japanese",
    "ko": "Korean", "fr": "French", "de": "German", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "id": "Indonesian",
    "th": "Thai", "ar": "Standard Arabic", "hi": "Hindi", "tr": "Turkish",
    "nl": "Dutch", "pl": "Polish", "uk": "Ukrainian", "sv": "Swedish",
}


# -------- Curated Voice Design preset packs ----------------------------------
# Each entry: (suffix, instruct_string, note_for_user)
ENGLISH_DESIGN_PACK = [
    ("female_young_british",   "female, young adult, british accent",
     "Female · Young · British"),
    ("female_young_american",  "female, young adult, american accent",
     "Female · Young · American"),
    ("female_middle_neutral",  "female, middle-aged, moderate pitch",
     "Female · Middle-aged · Neutral"),
    ("female_elderly_warm",    "female, elderly, low pitch",
     "Female · Elderly · Warm"),
    ("male_young_american",    "male, young adult, american accent",
     "Male · Young · American"),
    ("male_young_australian",  "male, young adult, australian accent",
     "Male · Young · Australian"),
    ("male_middle_british",    "male, middle-aged, low pitch, british accent",
     "Male · Middle-aged · British"),
    ("male_elderly_low",       "male, elderly, very low pitch",
     "Male · Elderly · Deep"),
    ("female_whisper",         "female, young adult, whisper",
     "Female · Whisper"),
]

CHINESE_DESIGN_PACK = [
    ("female_young",    "female, young adult, high pitch",     "女 · 青年 · 高音调"),
    ("female_middle",   "female, middle-aged, moderate pitch", "女 · 中年"),
    ("male_young",      "male, young adult",                   "男 · 青年"),
    ("male_middle_low", "male, middle-aged, low pitch",        "男 · 中年 · 低音调"),
    ("male_elderly",    "male, elderly",                       "男 · 老年"),
    ("dialect_sichuan", "四川话",                                "四川话"),
    ("dialect_dongbei", "东北话",                                "东北话"),
    ("dialect_shaanxi", "陕西话",                                "陕西话"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def estimate_gender(audio: np.ndarray, sr: int) -> tuple[str, float]:
    """Rough gender classification from median F0.

    Returns (label, median_f0). Threshold ~165 Hz works for most languages
    and is intentionally conservative — borderline cases get tagged "neutral".
    """
    try:
        import librosa
        f0 = librosa.yin(audio.astype(np.float32), fmin=70, fmax=400, sr=sr,
                         frame_length=2048)
        # YIN returns invalid values where unvoiced; drop NaN/inf and outliers
        f0 = f0[np.isfinite(f0)]
        if len(f0) < 10:
            return "neutral", 0.0
        med = float(np.median(f0))
        if med < 145:
            return "male", med
        if med > 175:
            return "female", med
        return "neutral", med
    except Exception as e:
        log.debug("Pitch estimation failed: %s", e)
        return "neutral", 0.0


def save_profile(name: str, audio: np.ndarray, sr: int, *,
                 ref_text: str, language: str, note: str,
                 tags: list[str], skip_existing: bool) -> bool:
    wav_path = VOICE_DIR / f"{name}.wav"
    json_path = VOICE_DIR / f"{name}.json"
    if skip_existing and wav_path.exists() and json_path.exists():
        log.info("[skip] %s already exists", name)
        return False
    sf.write(wav_path, audio, sr)
    json_path.write_text(
        json.dumps(
            {
                "ref_text": ref_text,
                "language": language,
                "note": note,
                "tags": tags,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    log.info("[ok]   %s  (%.2fs)  -> %s", name, len(audio) / sr, wav_path.name)
    return True


def gen_one(model: OmniVoice, text: str, language_name: Optional[str],
            instruct: Optional[str], num_step: int, seed: Optional[int],
            voice_clone_prompt=None) -> np.ndarray:
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    cfg = OmniVoiceGenerationConfig(num_step=num_step)
    kw = dict(text=text, language=language_name, generation_config=cfg)
    if instruct:
        kw["instruct"] = instruct
    if voice_clone_prompt is not None:
        kw["voice_clone_prompt"] = voice_clone_prompt
    with torch.inference_mode():
        out = model.generate(**kw)
    return out[0]


def cleanup_existing_pack(prefix: str):
    """Remove old profiles starting with `prefix` (used before regeneration)."""
    removed = 0
    for p in list(VOICE_DIR.glob(f"{prefix}*")):
        if p.suffix in {".wav", ".json", ".txt"}:
            p.unlink()
            removed += 1
    if removed:
        log.info("[clean] removed %d existing files for prefix '%s*'", removed, prefix)


XLINGUAL_DESIGN_PACK = [
    # (suffix, instruct_in_english, note_for_user)
    ("female_young",       "female, young adult",                 "Female · Young (xlingual)"),
    ("female_young_high",  "female, young adult, high pitch",     "Female · Young · High pitch"),
    ("female_middle",      "female, middle-aged, moderate pitch", "Female · Middle-aged"),
    ("female_elderly",     "female, elderly, low pitch",          "Female · Elderly · Warm"),
    ("male_young",         "male, young adult",                   "Male · Young (xlingual)"),
    ("male_middle",        "male, middle-aged, moderate pitch",   "Male · Middle-aged"),
    ("male_middle_low",    "male, middle-aged, low pitch",        "Male · Middle-aged · Deep"),
    ("male_elderly",       "male, elderly, very low pitch",       "Male · Elderly · Bass"),
]


# Chinese-source xlingual pack — yields Chinese-accented voices in the target
# language. Voice Design is well-trained on Chinese, so gender/age/pitch
# control is more reliable than via English.
# Each entry: (suffix, instruct_in_chinese, descriptive_note)
XLINGUAL_ZH_PACK = [
    # ---------- 10 Female (sorted low → high pitch within each age group) ----------
    ("female_elderly_deep",  "女, 老年, 低音调",      "Female · Elderly · Deep (低音调老婆婆)"),
    ("female_elderly",       "女, 老年, 中音调",      "Female · Elderly (老婆婆)"),
    ("female_middle_low",    "女, 中年, 低音调",      "Female · Mature · Deep (中年低音)"),
    ("female_middle",        "女, 中年, 中音调",      "Female · Middle-aged (中年女)"),
    ("female_young",         "女, 青年, 中音调",      "Female · Young · Neutral (青年女中音)"),
    ("female_young_high",    "女, 青年, 高音调",      "Female · Young · Bright (青年女高音)"),
    ("female_young_vhigh",   "女, 青年, 极高音调",    "Female · Young · Very high (极高音调)"),
    ("female_teen",          "女, 少年",              "Female · Teenager (少女)"),
    ("female_child",         "女, 儿童",              "Female · Child (女童)"),
    ("female_whisper",       "女, 青年, 耳语",        "Female · Young · Whisper (耳语)"),

    # ---------- 10 Male (sorted low → high pitch within each age group) -----------
    ("male_elderly_deep",    "男, 老年, 极低音调",    "Male · Elderly · Bass (老爷爷极低音)"),
    ("male_elderly",         "男, 老年, 低音调",      "Male · Elderly (老爷爷)"),
    ("male_middle_vlow",     "男, 中年, 极低音调",    "Male · Mature · Very deep (中年极低音)"),
    ("male_middle_low",      "男, 中年, 低音调",      "Male · Mature · Deep (中年低音)"),
    ("male_middle",          "男, 中年, 中音调",      "Male · Middle-aged (中年男)"),
    ("male_young_low",       "男, 青年, 低音调",      "Male · Young · Deep (青年男低音)"),
    ("male_young",           "男, 青年, 中音调",      "Male · Young · Neutral (青年男中音)"),
    ("male_young_high",      "男, 青年, 高音调",      "Male · Young · Bright (青年男高音)"),
    ("male_teen",            "男, 少年",              "Male · Teenager (少年)"),
    ("male_child",           "男, 儿童",              "Male · Child (男童)"),
]


# ---------------------------------------------------------------------------
# Per-language generation
# ---------------------------------------------------------------------------
def build_random_pack(model: OmniVoice, lang_id: str, *, n: int, num_step: int,
                      skip_existing: bool, base_seed: int):
    """Auto Voice × n. Auto-classify by pitch and rename to <lang>_<gender>_NN."""
    text = SAMPLE_TEXT.get(lang_id)
    lang_name = LANG_NAME_FOR_GENERATE.get(lang_id, lang_id)
    if not text:
        log.warning("No sample text for '%s' — skipping random pack", lang_id)
        return

    # If the user wants a fresh random pack, wipe the prior one so labels
    # stay consistent (the relative order changes when N changes).
    if not skip_existing:
        cleanup_existing_pack(f"gallery_{lang_id}_random_")
        # Also clean legacy "gallery_<lang>_NN" without 'random' segment
        for p in list(VOICE_DIR.glob(f"gallery_{lang_id}_[0-9][0-9].*")):
            p.unlink()

    samples = []   # list of (audio, gender, f0)
    log.info("Generating %d random voices for %s ...", n, lang_id)
    for i in range(n):
        try:
            audio = gen_one(model, text, lang_name, None, num_step,
                            seed=base_seed + i + 1)
        except Exception as e:
            log.error("Failed sample #%d: %s", i + 1, e)
            continue
        gender, f0 = estimate_gender(audio, model.sampling_rate)
        log.info("  · sample #%d → %s (median F0 = %.0f Hz)", i + 1, gender, f0)
        samples.append((audio, gender, f0))

    # Sort by gender → pitch (low to high)
    by_gender = {"female": [], "male": [], "neutral": []}
    for s in samples:
        by_gender[s[1]].append(s)
    for g in by_gender:
        by_gender[g].sort(key=lambda x: x[2])

    saved_count = 0
    for gender in ("female", "male", "neutral"):
        for j, (audio, _, f0) in enumerate(by_gender[gender], 1):
            name = f"gallery_{lang_id}_{gender}_{j:02d}"
            label = {"female": "Female", "male": "Male", "neutral": "Neutral"}[gender]
            note = f"{label} · {f0:.0f} Hz · auto"
            if save_profile(
                name, audio, model.sampling_rate,
                ref_text=text, language=lang_name, note=note,
                tags=["gallery", "random", lang_id, gender],
                skip_existing=False,
            ):
                saved_count += 1
    log.info("Saved %d random profiles for %s (female=%d, male=%d, neutral=%d)",
             saved_count, lang_id,
             len(by_gender["female"]), len(by_gender["male"]), len(by_gender["neutral"]))


def build_design_pack(model: OmniVoice, lang_id: str, pack: list[tuple[str, str, str]],
                      *, num_step: int, skip_existing: bool):
    """Voice Design × pack entries → save as gallery_<lang>_<suffix>."""
    text = SAMPLE_TEXT[lang_id]
    lang_name = LANG_NAME_FOR_GENERATE[lang_id]
    for suffix, instruct, note in pack:
        name = f"gallery_{lang_id}_{suffix}"
        if skip_existing and (VOICE_DIR / f"{name}.wav").exists():
            log.info("[skip] %s already exists", name)
            continue
        try:
            audio = gen_one(model, text, lang_name, instruct, num_step, seed=None)
        except Exception as e:
            log.error("Failed %s: %s", name, e)
            continue
        save_profile(
            name, audio, model.sampling_rate,
            ref_text=text,
            language=lang_name,
            note=note,
            tags=["gallery", "design", lang_id] + [s.strip() for s in instruct.split(",")],
            skip_existing=False,
        )


def build_xlingual_design_pack(model: OmniVoice, target_lang: str, *,
                                source_lang: str = "zh",
                                num_step: int, skip_existing: bool):
    """Two-step generation that yields *named* voices for non-EN/ZH languages.

    Step 1: Voice Design with source-language text → controlled reference audio.
    Step 2: Cross-lingual cloning of step-1 audio → audio in the target language.
    Step 3: Save the target-language audio as the gallery profile.

    The resulting voices carry an accent from the source language. Use
    ``source_lang="zh"`` (default) for a richer + more reliable design space
    (Voice Design is best-trained on Chinese), or ``"en"`` if you prefer
    English-accented output.

    Profiles are named ``gallery_<target>_xl_<suffix>`` and existing files
    with that prefix are removed before regeneration so old packs don't
    pollute the gallery.
    """
    if target_lang in {"en", "zh"}:
        log.info("Skipping xlingual pack for %s (use the native design pack)", target_lang)
        return
    target_text = SAMPLE_TEXT.get(target_lang)
    target_lang_name = LANG_NAME_FOR_GENERATE.get(target_lang)
    if not target_text or not target_lang_name:
        log.warning("Cannot build xlingual pack for unknown language '%s'", target_lang)
        return
    if source_lang not in {"en", "zh"}:
        raise ValueError("source_lang must be 'en' or 'zh'")

    pack = XLINGUAL_ZH_PACK if source_lang == "zh" else XLINGUAL_DESIGN_PACK
    src_text = SAMPLE_TEXT[source_lang]
    src_lang_name = LANG_NAME_FOR_GENERATE[source_lang]

    # Wipe old xl profiles so the user doesn't end up with a confusing mix
    if not skip_existing:
        cleanup_existing_pack(f"gallery_{target_lang}_xl_")

    log.info("Building cross-lingual design pack for %s (%d entries, source=%s)",
             target_lang, len(pack), source_lang)

    for suffix, instruct, note in pack:
        name = f"gallery_{target_lang}_xl_{suffix}"
        if skip_existing and (VOICE_DIR / f"{name}.wav").exists():
            log.info("[skip] %s already exists", name)
            continue
        try:
            # Step 1: controlled reference in source language
            log.info("  · %s — generating %s reference [%s] ...",
                     name, src_lang_name, instruct)
            src_audio = gen_one(model, src_text, src_lang_name, instruct, num_step, seed=None)

            # Step 2: cross-lingual clone
            prompt = model.create_voice_clone_prompt(
                ref_audio=(torch.from_numpy(src_audio).unsqueeze(0), model.sampling_rate),
                ref_text=src_text,
                preprocess_prompt=True,
            )
            log.info("    cross-lingual clone %s → %s ...", src_lang_name, target_lang_name)
            target_audio = gen_one(
                model, target_text, target_lang_name, None, num_step, seed=None,
                voice_clone_prompt=prompt,
            )
        except Exception as e:
            log.error("Failed %s: %s", name, e)
            continue

        save_profile(
            name, target_audio, model.sampling_rate,
            ref_text=target_text,
            language=target_lang_name,
            note=note,
            tags=["gallery", "xlingual", target_lang, f"src:{source_lang}"]
                 + [s.strip() for s in instruct.split(",")],
            skip_existing=False,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_languages(s: str) -> list[str]:
    return [tok.strip().lower() for tok in s.split(",") if tok.strip()]


def main():
    p = argparse.ArgumentParser(description="Pre-build OmniVoice voice gallery")
    p.add_argument("--model", default="k2-fsa/OmniVoice")
    p.add_argument("--device", default=None,
                   help="cuda | cpu | mps  (auto-detected if omitted)")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "float32", "bfloat16"])
    p.add_argument("--languages",
                   default="vi,en,zh",
                   help="Comma-separated list of language codes (default: vi,en,zh)")
    p.add_argument("--random-voices", type=int, default=4,
                   help="Random voices per non-EN/ZH language (default: 4)")
    p.add_argument("--num-step", type=int, default=24,
                   help="Diffusion steps when generating samples (default: 24)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Don't regenerate profiles already on disk")
    p.add_argument("--no-design", action="store_true",
                   help="Don't build the EN/ZH design preset packs")
    p.add_argument("--no-random", action="store_true",
                   help="Don't build random packs for non-EN/ZH languages")
    p.add_argument("--xlingual", action="store_true",
                   help="Also build a cross-lingual design pack for non-EN/ZH "
                        "languages (named voices like '<lang>_xl_female_young', "
                        "may carry a slight accent from the source language).")
    p.add_argument("--xlingual-via", choices=["en", "zh"], default="zh",
                   help="Source language for the cross-lingual pack. "
                        "'zh' (default) gives Chinese-accented voices with "
                        "20 entries (10 female + 10 male) covering child→elderly. "
                        "'en' gives English-accented voices with 8 entries.")
    p.add_argument("--base-seed", type=int, default=42)
    args = p.parse_args()

    # Pick device + dtype
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = args.device
    dtype = {"float16": torch.float16, "float32": torch.float32,
             "bfloat16": torch.bfloat16}[args.dtype]
    if device == "cpu" and dtype == torch.float16:
        dtype = torch.float32

    log.info("Loading OmniVoice on %s (%s) ...", device, args.dtype)
    t0 = time.time()
    model = OmniVoice.from_pretrained(args.model, device_map=device, dtype=dtype)
    log.info("Loaded in %.1fs", time.time() - t0)

    langs = parse_languages(args.languages)

    overall_t0 = time.time()
    total_count = 0
    for lang in langs:
        if lang not in SAMPLE_TEXT:
            log.warning("'%s' not in built-in sample-text table; skipping", lang)
            continue
        log.info("== Language: %s (%s) ==", lang, LANG_NAME_FOR_GENERATE[lang])
        before = len(list(VOICE_DIR.glob(f"gallery_{lang}_*.wav")))

        # Design pack for EN & ZH
        if not args.no_design and lang == "en":
            build_design_pack(model, "en", ENGLISH_DESIGN_PACK,
                              num_step=args.num_step, skip_existing=args.skip_existing)
        elif not args.no_design and lang == "zh":
            build_design_pack(model, "zh", CHINESE_DESIGN_PACK,
                              num_step=args.num_step, skip_existing=args.skip_existing)

        # Cross-lingual design pack for non-EN/ZH (opt-in via --xlingual)
        if args.xlingual and lang not in {"en", "zh"}:
            build_xlingual_design_pack(
                model, lang,
                source_lang=args.xlingual_via,
                num_step=args.num_step, skip_existing=args.skip_existing,
            )

        # Random pack for everything else (and as bonus for EN/ZH if requested)
        if not args.no_random:
            build_random_pack(
                model, lang, n=args.random_voices,
                num_step=args.num_step,
                skip_existing=args.skip_existing,
                base_seed=args.base_seed,
            )
        after = len(list(VOICE_DIR.glob(f"gallery_{lang}_*.wav")))
        added = after - before
        total_count += added
        log.info("   -> %d gallery profile(s) for %s (added %d)", after, lang, added)

    log.info("Done. Generated %d new profile(s) in %.1fs",
             total_count, time.time() - overall_t0)
    log.info("Gallery directory: %s", VOICE_DIR)


if __name__ == "__main__":
    main()
