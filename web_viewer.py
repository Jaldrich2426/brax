import http.server
import socketserver
import sys
import os

# Check if an HTML file was provided as a command-line argument
if len(sys.argv) != 2:
    print("Usage: python serve_html.py <path_to_html_file>")
    sys.exit(1)

# Get the path to the HTML file
html_file_path = sys.argv[1]

# Ensure the file exists
if not os.path.isfile(html_file_path):
    print(f"File not found: {html_file_path}")
    sys.exit(1)

# Define the handler to serve the file
class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.path = html_file_path
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

# Set the server address and port
PORT = 8888
Handler = MyHttpRequestHandler

# Create the server object
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.allow_reuse_address = True  # Allow socket reuse
    print(f"Serving {html_file_path} at port {PORT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer is shutting down...")
        httpd.server_close()
        print("Server has shut down")