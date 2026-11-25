"""Loopback HTTP integration tests; uses synthetic vision, never a real camera."""
import json
import threading
import unittest
from functools import partial
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from pulselens.session import Session
from pulselens.web import Registry, handler_for
from test_python_pipeline import FakeVision


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = Registry(partial(Session, vision_factory=FakeVision))
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(cls.registry))
        cls.server.daemon_threads = True
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True); cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join(); cls.registry.close()

    def request(self, path, data=None, headers=None):
        request = Request(self.base + path, data=data, headers=headers or {})
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, response.read(), response.headers
        except HTTPError as error:
            return error.code, error.read(), error.headers

    def test_health_and_static_interface(self):
        status, body, _ = self.request('/api/health')
        self.assertEqual(status, 200); self.assertEqual(json.loads(body)['backend'], 'python')
        status, body, _ = self.request('/')
        self.assertEqual(status, 200); self.assertIn(b'Research insights', body)
        status, _, _ = self.request('/src/ui-utils.js'); self.assertEqual(status, 200)

    def test_camera_consent_and_frame_pipeline(self):
        status, _, _ = self.request('/api/start', b'{"method":"pos","consent":false}')
        self.assertEqual(status, 400)
        status, body, _ = self.request('/api/start', b'{"method":"pos","consent":true}')
        self.assertEqual(status, 200); token = json.loads(body)['session']
        headers = {'X-Session-Id': token, 'X-Frame-Width': '36', 'X-Frame-Height': '36', 'X-Frame-Time': '1000'}
        status, body, response_headers = self.request('/api/frame', bytes([128, 128, 128, 255]) * (36 * 36), headers)
        self.assertEqual(status, 200); result = json.loads(body)
        self.assertTrue(result['face']); self.assertEqual(result['capture']['samples'], 1)
        self.assertEqual(response_headers['Cache-Control'], 'no-store')
        status, _, _ = self.request('/api/stop', b'', {'X-Session-Id': token})
        self.assertEqual(status, 200); self.assertNotIn(token, self.registry.sessions)

    def test_no_private_source_or_cross_origin_frames(self):
        for path in ['/pulselens/assets.py', '/.git/config', '/requirements.txt', '/models/state.gz', '/src/deep.worker.js']:
            status, _, _ = self.request(path); self.assertEqual(status, 404)
        status, _, _ = self.request('/api/start', b'{}', {'Origin': 'https://example.org'})
        self.assertEqual(status, 403)

    def test_invalid_session_and_method(self):
        status, _, _ = self.request('/api/frame', b'', {'X-Session-Id': 'expired'})
        self.assertEqual(status, 409)
        status, _, _ = self.request('/api/start', b'{"method":"unknown","consent":true}')
        self.assertEqual(status, 400)

if __name__ == '__main__':
    unittest.main()
