from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import threading
import wave
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

APP_DIR = Path(__file__).resolve().parent

def get_cache_root() -> Path:
    configured = os.environ.get('WHISPER_DICTATION_CACHE')
    if configured:
        return Path(configured).expanduser()
    return Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))) / 'whisper-dictation'


CACHE_ROOT = get_cache_root()
WHISPER_DIR = Path(os.environ.get('WHISPER_CPP_DIR', str(CACHE_ROOT / 'whisper.cpp'))).expanduser()
MODEL_DIR = Path(os.environ.get('WHISPER_MODEL_DIR', str(WHISPER_DIR / 'models'))).expanduser()
CLI_PATH = Path(os.environ.get('WHISPER_CLI_PATH', str(WHISPER_DIR / 'build-openvino/bin/whisper-cli'))).expanduser()
MODEL_PATH = MODEL_DIR / 'ggml-base.bin'
SAMPLE_PATH = APP_DIR.parent / 'sample_phrase.wav'
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.ogg', '.webm'}
SUPPORTED_LANGUAGE_MODES = {'auto', 'en', 'ru'}
TRANSCRIPTION_LOCK = threading.Lock()


def transcribe_audio(audio_bytes: bytes, suffix: str, language_mode: str) -> dict[str, object]:
    if not audio_bytes:
        raise ValueError('The audio file is empty.')
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError('Audio files must be 25 MB or smaller.')
    if language_mode not in SUPPORTED_LANGUAGE_MODES:
        raise ValueError('Choose Auto, English, or Russian for the language setting.')

    with tempfile.TemporaryDirectory(prefix='whisper-dictation-') as temp_dir:
        temp = Path(temp_dir)
        source = temp / ('upload' + suffix)
        audio_wav = temp / 'audio.wav'
        output_base = temp / 'transcript'
        source.write_bytes(audio_bytes)

        conversion = subprocess.run(
            ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
             '-i', str(source), '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(audio_wav)],
            capture_output=True, text=True, errors='replace', timeout=120,
        )
        if conversion.returncode != 0:
            detail = conversion.stderr.strip()[-1000:]
            raise ValueError('Could not read this audio recording.' + (f' {detail}' if detail else ''))

        language_arg = 'auto' if language_mode == 'auto' else language_mode
        command = [
            str(CLI_PATH), '-m', str(MODEL_PATH), '-f', str(audio_wav),
            '-l', language_arg, '-bs', '5', '-t', '4', '-oved', 'GPU',
            '-nt', '-otxt', '-of', str(output_base),
        ]
        env = os.environ.copy()
        env['ONEAPI_DEVICE_SELECTOR'] = 'opencl:gpu'
        with TRANSCRIPTION_LOCK:
            result = subprocess.run(
                command, cwd=WHISPER_DIR, env=env, capture_output=True,
                text=True, errors='replace', timeout=600,
            )
        combined = result.stdout + '\n' + result.stderr
        if result.returncode != 0:
            print('OpenVINO transcription failed:\n' + combined[-4000:], flush=True)
            raise RuntimeError('Transcription failed. Check the server terminal for details.')

        transcript_path = output_base.with_suffix('.txt')
        transcript = transcript_path.read_text(encoding='utf-8', errors='replace').strip() if transcript_path.exists() else ''
        detected = re.search(r'auto-detected language:\s*([a-z]{2})', combined, re.IGNORECASE)
        language = language_arg if language_arg != 'auto' else (detected.group(1).lower() if detected else None)
        with wave.open(str(audio_wav), 'rb') as audio_info:
            duration = audio_info.getnframes() / float(audio_info.getframerate())

        return {
            'text': transcript,
            'duration': round(duration, 2),
            'language': language,
            'language_mode': language_mode,
            'model': 'base · OpenVINO GPU',
        }


class DictationHandler(BaseHTTPRequestHandler):
    server_version = 'WhisperDictationOpenVINO/0.1'

    def do_GET(self) -> None:
        if urlparse(self.path).path not in {'/', '/index.html'}:
            self.send_error(404, 'Not found.')
            return
        try:
            page = (APP_DIR / 'index.html').read_bytes()
        except OSError:
            self.send_error(500, 'The app page could not be read.')
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(page)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(page)

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        if route == '/sample':
            self.handle_sample()
        elif route == '/transcribe':
            self.handle_upload()
        else:
            self.send_error(404, 'Not found.')

    def handle_sample(self) -> None:
        try:
            self.discard_body()
            query = parse_qs(urlparse(self.path).query)
            if query.get('model', ['base'])[0] != 'base':
                raise ValueError('This OpenVINO build includes the base model only.')
            language = query.get('language', ['auto'])[0]
            result = transcribe_audio(SAMPLE_PATH.read_bytes(), '.wav', language)
            self.send_json(200, result)
        except ValueError as error:
            self.send_json(400, {'error': str(error)})
        except Exception as error:
            print(f'Sample transcription failed: {error}', flush=True)
            self.send_json(500, {'error': 'Transcription failed. Check the server terminal for details.'})

    def handle_upload(self) -> None:
        try:
            audio_bytes, suffix, language = self.read_upload()
            self.send_json(200, transcribe_audio(audio_bytes, suffix, language))
        except ValueError as error:
            self.send_json(400, {'error': str(error)})
        except Exception as error:
            print(f'Transcription failed: {error}', flush=True)
            self.send_json(500, {'error': 'Transcription failed. Check the server terminal for details.'})

    def read_upload(self) -> tuple[bytes, str, str]:
        length = self.content_length()
        if length > MAX_UPLOAD_BYTES + 64 * 1024:
            raise ValueError('Audio files must be 25 MB or smaller.')
        content_type = self.headers.get('Content-Type', '')
        if not content_type.lower().startswith('multipart/form-data'):
            raise ValueError('Send an audio file using the upload form.')
        payload = self.rfile.read(length)
        raw = f'Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n'.encode() + payload
        message = BytesParser(policy=policy.default).parsebytes(raw)
        if not message.is_multipart():
            raise ValueError('The uploaded form could not be read.')
        language = 'auto'
        for part in message.iter_parts():
            name = part.get_param('name', header='content-disposition')
            if name == 'language':
                language = (part.get_payload(decode=True) or b'').decode('utf-8').strip()
                if language not in SUPPORTED_LANGUAGE_MODES:
                    raise ValueError('Choose Auto, English, or Russian for the language setting.')
            elif name == 'model':
                model = (part.get_payload(decode=True) or b'').decode('utf-8').strip()
                if model != 'base':
                    raise ValueError('This OpenVINO trial currently includes the base model only.')
            elif name == 'audio':
                filename = part.get_filename() or 'recording.wav'
                suffix = Path(filename.replace('\\', '/')).suffix.lower()
                if suffix not in ALLOWED_EXTENSIONS:
                    raise ValueError('Choose a WAV, MP3, M4A, OGG, or WebM audio file.')
                data = part.get_payload(decode=True) or b''
                if len(data) > MAX_UPLOAD_BYTES:
                    raise ValueError('Audio files must be 25 MB or smaller.')
                return data, suffix, language
        raise ValueError('Choose an audio file before transcribing.')

    def discard_body(self) -> None:
        length = self.content_length()
        if length:
            self.rfile.read(length)

    def content_length(self) -> int:
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError as error:
            raise ValueError('The request size is invalid.') from error
        if length < 0:
            raise ValueError('The request size is invalid.')
        return length

    def send_json(self, status: int, data: dict[str, object]) -> None:
        response = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(response)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format_string: str, *args: object) -> None:
        print(f'[{self.log_date_time_string()}] {format_string % args}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the WSL OpenVINO GPU dictation trial.')
    parser.add_argument('--port', type=int, default=8001)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    if not CLI_PATH.is_file() or not MODEL_PATH.is_file():
        parser.error('Build the OpenVINO whisper-cli and download ggml-base.bin first.')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), DictationHandler)
    print(f'Ready: http://127.0.0.1:{args.port} (base, OpenVINO GPU encoder, CPU decoder)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('Stopping the OpenVINO dictation trial.', flush=True)
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
