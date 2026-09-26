from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from faster_whisper import WhisperModel


APP_DIR = Path(__file__).resolve().parent

def get_cache_dir() -> Path:
    configured = os.environ.get("WHISPER_DICTATION_CACHE")
    if configured:
        return Path(configured).expanduser()
    if os.name == "nt":
        cache_root = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    else:
        cache_root = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return cache_root / "whisper-dictation"


MODEL_DIR = get_cache_dir() / "models"
SAMPLE_PATH = APP_DIR / "sample_phrase.wav"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".webm"}
SUPPORTED_MODELS = {"base", "small"}
SUPPORTED_LANGUAGE_MODES = {"auto", "en", "ru"}
DEFAULT_MODEL = "base"
DEFAULT_LANGUAGE_MODE = "auto"
TRANSCRIPTION_LOCK = threading.Lock()
MODEL_LOAD_LOCK = threading.Lock()
MODEL_CACHE: dict[str, WhisperModel] = {}


def get_model(model_name: str) -> WhisperModel:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError("Choose the base or small multilingual model.")
    model = MODEL_CACHE.get(model_name)
    if model is not None:
        return model
    with MODEL_LOAD_LOCK:
        model = MODEL_CACHE.get(model_name)
        if model is None:
            print(f"Loading Faster Whisper {model_name} on CPU (int8)...", flush=True)
            model = WhisperModel(
                model_name,
                device="cpu",
                compute_type="int8",
                download_root=str(MODEL_DIR),
            )
            MODEL_CACHE[model_name] = model
    return model


def transcribe_audio(
    audio_bytes: bytes,
    suffix: str,
    model_name: str,
    language_mode: str,
) -> dict[str, object]:
    if not audio_bytes:
        raise ValueError("The audio file is empty.")
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("Audio files must be 25 MB or smaller.")
    if language_mode not in SUPPORTED_LANGUAGE_MODES:
        raise ValueError("Choose Auto, English, or Russian for the language setting.")

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as audio_file:
            audio_file.write(audio_bytes)
            temp_path = audio_file.name

        model = get_model(model_name)
        with TRANSCRIPTION_LOCK:
            segments, info = model.transcribe(
                temp_path,
                language=None if language_mode == "auto" else language_mode,
                beam_size=5,
                vad_filter=True,
                multilingual=language_mode == "auto",
            )
            transcript = "".join(segment.text for segment in segments).strip()

        return {
            "text": transcript,
            "duration": round(float(info.duration), 2),
            "language": info.language,
            "language_mode": language_mode,
            "model": model_name,
        }
    finally:
        if temp_path is not None:
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass


class DictationHandler(BaseHTTPRequestHandler):
    server_version = "WhisperDictation/1.0"

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route in {"/", "/index.html"}:
            try:
                page = (APP_DIR / "index.html").read_bytes()
            except OSError:
                self.send_error(500, "The app page could not be read.")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(page)
            return

        self.send_error(404, "Not found.")

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        if route == "/sample":
            self._handle_sample()
        elif route == "/transcribe":
            self._handle_upload()
        else:
            self.send_error(404, "Not found.")

    def _handle_sample(self) -> None:
        try:
            self._check_empty_request()
            audio_bytes = SAMPLE_PATH.read_bytes()
            query = parse_qs(urlparse(self.path).query)
            model_name = query.get("model", [DEFAULT_MODEL])[0]
            language_mode = query.get("language", [DEFAULT_LANGUAGE_MODE])[0]
            result = transcribe_audio(audio_bytes, ".wav", model_name, language_mode)
            self._send_json(200, result)
        except ValueError as error:
            self._send_json(400, {"error": str(error)})
        except OSError:
            self._send_json(500, {"error": "The sample audio file could not be read."})
        except Exception as error:
            print(f"Sample transcription failed: {error}", flush=True)
            self._send_json(500, {"error": "Transcription failed. Check the server terminal for details."})

    def _handle_upload(self) -> None:
        try:
            audio_bytes, suffix, model_name, language_mode = self._read_uploaded_audio()
            result = transcribe_audio(audio_bytes, suffix, model_name, language_mode)
            self._send_json(200, result)
        except ValueError as error:
            self._send_json(400, {"error": str(error)})
        except Exception as error:
            print(f"Transcription failed: {error}", flush=True)
            self._send_json(500, {"error": "Transcription failed. Check the server terminal for details."})

    def _read_uploaded_audio(self) -> tuple[bytes, str, str, str]:
        content_length = self._content_length()
        if content_length > MAX_UPLOAD_BYTES + 64 * 1024:
            raise ValueError("Audio files must be 25 MB or smaller.")

        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data"):
            raise ValueError("Send an audio file using the upload form.")

        payload = self.rfile.read(content_length)
        mime_message = (
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
            + payload
        )
        message = BytesParser(policy=policy.default).parsebytes(mime_message)
        if not message.is_multipart():
            raise ValueError("The uploaded form could not be read.")

        model_name = DEFAULT_MODEL
        language_mode = DEFAULT_LANGUAGE_MODE
        for part in message.iter_parts():
            field_name = part.get_param("name", header="content-disposition")
            if field_name == "model":
                model_name = (part.get_payload(decode=True) or b"").decode("utf-8").strip()
                if model_name not in SUPPORTED_MODELS:
                    raise ValueError("Choose the base or small multilingual model.")
                continue
            if field_name == "language":
                language_mode = (part.get_payload(decode=True) or b"").decode("utf-8").strip()
                if language_mode not in SUPPORTED_LANGUAGE_MODES:
                    raise ValueError("Choose Auto, English, or Russian for the language setting.")
                continue
            if field_name != "audio":
                continue
            filename = part.get_filename() or "recording.wav"
            suffix = Path(filename.replace("\\", "/")).suffix.lower()
            if suffix not in ALLOWED_EXTENSIONS:
                raise ValueError("Choose a WAV, MP3, M4A, OGG, or WebM audio file.")
            audio_bytes = part.get_payload(decode=True) or b""
            if len(audio_bytes) > MAX_UPLOAD_BYTES:
                raise ValueError("Audio files must be 25 MB or smaller.")
            return audio_bytes, suffix, model_name, language_mode

        raise ValueError("Choose an audio file before transcribing.")

    def _check_empty_request(self) -> None:
        content_length = self._content_length()
        if content_length:
            self.rfile.read(content_length)

    def _content_length(self) -> int:
        value = self.headers.get("Content-Length", "0")
        try:
            length = int(value)
        except ValueError as error:
            raise ValueError("The request size is invalid.") from error
        if length < 0:
            raise ValueError("The request size is invalid.")
        return length

    def _send_json(self, status: int, data: dict[str, object]) -> None:
        response = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format_string: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format_string % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Faster Whisper dictation.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    get_model(DEFAULT_MODEL)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), DictationHandler)
    print(f"Ready: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping local dictation server.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
