import http.server
import socketserver
import sys
import requests

# Check if a URL was provided as a command-line argument
if len(sys.argv) != 2:
    print("Usage: python serve_html.py <url>")
    sys.exit(1)

# Get the URL
url = sys.argv[1]

# Fetch the HTML content from the URL
try:
    response = requests.get(url)
    response.raise_for_status()
    html_content = response.text
except requests.exceptions.RequestException as e:
    print(f"Error fetching URL: {e}")
    sys.exit(1)

# Define the handler to serve the content
class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            # Serve the fetched HTML content
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(html_content.encode('utf-8'))
        else:
            self.send_error(404, "File not found")

# Set the server address and port
PORT = 8888
Handler = MyHttpRequestHandler

# Create the server object
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.allow_reuse_address = True  # Allow socket reuse
    print(f"Serving content from {url} at port {PORT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer is shutting down...")
        httpd.server_close()
        print("Server has shut down")
