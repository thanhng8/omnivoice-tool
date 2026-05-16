# Voice Profiles

Drop reference audio files here to use them as cloned voices. The server
auto-scans this folder on startup and after each upload via the web UI.

## Expected layout

```
voice_prompts/
├── alice.wav        # 3–10s reference audio (any sample rate; resampled internally)
├── alice.json       # {"ref_text": "...", "language": "English", "note": "..."}
├── bob.wav
├── bob.txt          # plain transcript (alternative to .json's "ref_text")
└── ...
```

`name.json` is preferred. `name.txt` is supported for compatibility with the
supertonic-style folder layout (its content becomes `ref_text`).

If neither metadata file exists, the server will still load the profile but
will need Whisper ASR to auto-transcribe at synthesis time (slower; can be
disabled with `--no-asr`).

## Tips

- Use a clean, mono recording. Trim silences. Avoid background music.
- Match the language of the reference to the synthesised text for cleanest
  pronunciation. Cross-lingual cloning works but carries the source accent.
- Filenames are sluggified server-side: only `[A-Za-z0-9_-]` are kept.
