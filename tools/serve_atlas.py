#!/usr/bin/env python3
import http.server
import socketserver
import os

PORT = 8008
DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "atlas")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)


print(f"Serving Structure Atlas Viewer at http://localhost:{PORT}")
print("Press Ctrl+C to stop.")

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
