import threading
from http.server import HTTPServer, BaseHTTPRequestHandler


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()


def startHealth():
    server = HTTPServer(("0.0.0.0", 8080), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
