"""Loopback-only HTTP server for the Python measurement pipeline."""
import argparse
import json
import mimetypes
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
import numpy as np
from .assets import ROOT
from .session import Session

PUBLIC = {'/', '/index.html', '/favicon.svg', '/manifest.webmanifest', '/THIRD_PARTY_NOTICES.md',
          '/src/app.js', '/src/camera.js', '/src/charts.js', '/src/visuals.js', '/src/insights.js',
          '/src/ui-utils.js', '/src/style.css', '/src/insights.css'}
MAX_BODY = 640 * 480 * 4


def decode_frame(body, width, height):
    if width < 36 or height < 36 or width > 640 or height > 480 or len(body) != width * height * 4:
        raise ValueError('Expected an RGBA camera frame no larger than 640 by 480.')
    return np.frombuffer(body, np.uint8).reshape(height, width, 4)[:, :, :3].copy()


class Registry:
    def __init__(self, factory=Session):
        self.factory = factory; self.sessions = {}; self.lock = threading.RLock(); self.pending = False

    def start(self, method):
        with self.lock:
            for token, session in list(self.sessions.items()):
                if time.monotonic() - session.last_used > 120 and session.lock.acquire(blocking=False):
                    try:
                        session.close(); del self.sessions[token]
                    finally:
                        session.lock.release()
            if self.pending or len(self.sessions) >= 2:
                raise ValueError('Close another measurement session before starting a new one.')
            self.pending = True
        try:
            session = self.factory(method)
            token = secrets.token_urlsafe(32)
            with self.lock:
                self.sessions[token] = session
            return token
        finally:
            with self.lock:
                self.pending = False

    def get(self, token):
        with self.lock:
            if token not in self.sessions:
                raise KeyError('Session expired. Start a new session.')
            return self.sessions[token]

    def stop(self, token):
        with self.lock:
            session = self.sessions.pop(token, None)
        if session:
            with session.lock:
                session.close()

    def close(self):
        for token in list(self.sessions):
            self.stop(token)


def model_check():
    from .facephys import FacePhys
    from .vision import FaceVision
    model = FacePhys(); pulse = []
    for method in ('pos', 'deep'):
        detector = FaceVision(method)
        try:
            detector.process(np.full((480, 640, 3), 128, np.uint8), 1000)
        finally:
            detector.close()
    for _ in range(480):
        value, _ = model.step(np.full((36, 36, 3), 128, np.uint8), 1 / 30)
        pulse.append(value)
    result = model.analyze(pulse[-450:], 1 / 30)
    return dict(ok=True, frames=len(pulse), inference=True, sqi=bool(np.isfinite(result['quality'])), psd=True)


def handler_for(registry):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Do not log frames, tokens, timestamps, or physiological values.

        def allowed_origin(self):
            host = self.headers.get('Host', '').split(':')[0]
            if host not in ('localhost', '127.0.0.1'):
                return False
            origin = self.headers.get('Origin')
            return not origin or origin == 'http://' + self.headers.get('Host', '')

        def send_json(self, status, data):
            payload = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers(); self.wfile.write(payload)

        def do_GET(self):
            if not self.allowed_origin():
                return self.send_json(403, dict(error='Use this app on localhost.'))
            path = urlsplit(self.path).path
            if path == '/api/health':
                return self.send_json(200, dict(ok=True, backend='python', inference='local-server'))
            if path not in PUBLIC:
                return self.send_json(404, dict(error='Not found.'))
            target = ROOT / ('index.html' if path == '/' else path.lstrip('/'))
            try:
                content = target.read_bytes()
            except OSError:
                return self.send_json(404, dict(error='Static asset unavailable.'))
            self.send_response(200)
            self.send_header('Content-Type', mimetypes.guess_type(target.name)[0] or 'application/octet-stream')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers(); self.wfile.write(content)

        def do_POST(self):
            if not self.allowed_origin():
                return self.send_json(403, dict(error='Cross-origin camera requests are blocked.'))
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if length < 0 or length > MAX_BODY:
                    return self.send_json(413, dict(error='Frame/request too large.'))
                body = self.rfile.read(length)
                path = urlsplit(self.path).path
                if path == '/api/start':
                    data = json.loads(body)
                    if data.get('consent') is not True:
                        return self.send_json(400, dict(error='Local processing consent is required.'))
                    token = registry.start(data.get('method'))
                    return self.send_json(200, dict(session=token, backend='python'))
                if path == '/api/stop':
                    registry.stop(self.headers.get('X-Session-Id', ''))
                    return self.send_json(200, dict(ok=True))
                if path == '/api/frame':
                    session = registry.get(self.headers.get('X-Session-Id', ''))
                    image = decode_frame(body, int(self.headers.get('X-Frame-Width', '0')), int(self.headers.get('X-Frame-Height', '0')))
                    with session.lock:
                        result = session.process(image, float(self.headers['X-Frame-Time']),
                            self.headers.get('X-Amplify') == 'true', float(self.headers.get('X-Gain', '10')))
                    return self.send_json(200, result)
                if path == '/api/check':
                    return self.send_json(200, model_check())
                return self.send_json(404, dict(error='Unknown API route.'))
            except KeyError as error:
                self.send_json(409, dict(error=str(error)))
            except (ValueError, TypeError) as error:
                self.send_json(400, dict(error=str(error)))
            except Exception as error:
                self.send_json(503, dict(error=f'Python processing error: {error}'))
    return Handler


def main():
    parser = argparse.ArgumentParser(description='PulseLens Python website')
    parser.add_argument('--port', type=int, default=4173)
    args = parser.parse_args()
    registry = Registry()
    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(registry))
    except OSError as error:
        raise SystemExit(f'Cannot open port {args.port}: {error}. Stop the old npm server or use --port 4174.')
    server.daemon_threads = True
    print(f'PulseLens Python ready: http://localhost:{args.port}', flush=True)
    print('Frames are processed on this computer only. Press Ctrl+C to stop.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown(); registry.close(); server.server_close()
