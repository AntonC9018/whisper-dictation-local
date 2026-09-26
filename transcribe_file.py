from __future__ import annotations

import argparse

from faster_whisper import WhisperModel

from server import MODEL_DIR, SUPPORTED_LANGUAGE_MODES, SUPPORTED_MODELS


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe an audio file with Faster Whisper.")
    parser.add_argument("audio", help="Path to a supported audio file")
    parser.add_argument("--model", choices=sorted(SUPPORTED_MODELS), default="base")
    parser.add_argument("--language", choices=sorted(SUPPORTED_LANGUAGE_MODES), default="auto")
    args = parser.parse_args()

    model = WhisperModel(
        args.model,
        device="cpu",
        compute_type="int8",
        download_root=str(MODEL_DIR),
    )
    segments, info = model.transcribe(
        args.audio,
        language=None if args.language == "auto" else args.language,
        beam_size=5,
        vad_filter=True,
        multilingual=args.language == "auto",
    )
    print("".join(segment.text for segment in segments).strip())
    print(f"Detected language: {info.language}")
    print(f"Audio length: {info.duration:.2f} seconds")


if __name__ == "__main__":
    main()
