#!/bin/bash
set -e
cd "$(dirname "$0")"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

detect_python() {
    if command -v omnivoice-infer &>/dev/null; then
        python -c "import sys, omnivoice; print(sys.executable)"
    elif command -v python3 &>/dev/null; then
        echo "python3"
    else
        echo "python"
    fi
}

PY="$(detect_python)"

if [ -n "$1" ]; then
    ARGS="$@"
    run_server
fi

show_menu() {
    cls=clear
    command -v clear &>/dev/null || cls='echo -e "\n\n\n"'
    $cls 2>/dev/null || true

    echo "============================================================"
    echo "  OmniVoice TTS Server - Launcher"
    echo "============================================================"
    echo ""
    echo "  Web UI:  http://127.0.0.1:8765/  (default port)"
    echo "  Python:  $PY"
    echo ""
    echo "  Choose a launch mode:"
    echo ""
    echo "    [1] GPU + Whisper ASR  (default - uses ~4GB VRAM)"
    echo "        Allows uploading voice profiles without ref_text"
    echo "        (Whisper auto-transcribes the reference audio)"
    echo ""
    echo "    [2] GPU only (no Whisper) - saves 1.5GB VRAM"
    echo "        ref_text is required when uploading voice profiles"
    echo ""
    echo "    [3] CPU only  (slow - for machines without a GPU)"
    echo ""
    echo "    [4] GPU + Whisper, custom port"
    echo ""
    echo "    [5] Custom args (advanced)"
    echo ""
    echo "    [Q] Quit"
    echo ""
}

run_server() {
    if [ -z "$cls" ]; then
        if command -v clear &>/dev/null; then clear; else echo -e "\n\n"; fi
    fi

    echo "============================================================"
    echo "  OmniVoice TTS Server (HTTP + WebSocket)"
    echo "============================================================"
    echo ""
    echo "  Args:           $ARGS"
    echo "  Python:          $PY"
    echo "  HF_HUB_OFFLINE: $HF_HUB_OFFLINE"
    echo ""
    echo "  Web UI:          http://127.0.0.1:8765/"
    echo "  WebSocket:       ws://127.0.0.1:8765/ws"
    echo "  Voice profiles: $(pwd)/voice_prompts/"
    echo ""
    echo "  Press Ctrl+C to stop the server."
    echo "============================================================"
    echo ""

    "$PY" "$(dirname "$0")/ws_omnivoice_server.py" $ARGS
    echo ""
    read -p "Server stopped. Press Enter to exit..."
    exit 0
}

while true; do
    show_menu
    read -p "Your choice [1/2/3/4/5/Q]: " CHOICE

    case "$CHOICE" in
    1)  ARGS="";         run_server ;;
    2)  ARGS="--no-asr"; run_server ;;
    3)  ARGS="--cpu";    run_server ;;
    4)  read -p "Enter port [e.g. 8888]: " PORT
        ARGS="--port ${PORT:-8765}"
        run_server ;;
    5)  echo ""
        echo "Common flags:"
        echo "  --port N        Change port (default 8765)"
        echo "  --no-asr        Skip Whisper ASR"
        echo "  --cpu           Force CPU"
        echo "  --device cuda|mps|cpu   Pick a device"
        echo "  --dtype float16|float32|bfloat16"
        echo ""
        read -p "Enter args: " CUSTOM
        ARGS="$CUSTOM"
        run_server ;;
    Q|q) exit 0 ;;
    *)  echo "Invalid choice. Try again..."
        sleep 2 ;;
    esac
done