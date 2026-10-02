"""Local server for the offline watermarking playground.

Serves this folder with the cross-origin isolation headers WebGPU / threaded
WASM need (SharedArrayBuffer). Open http://127.0.0.1:8765/ — do not open
index.html as a file.
"""

from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import ctypes
import mimetypes
import os
from ctypes import wintypes

PORT = 8765

mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/octet-stream", ".onnx")


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Required for crossOriginIsolated (multi-thread WASM, WebGPU in workers).
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args):
        # Skip noisy progress on multi-gigabyte weight files.
        path = self.path.split("?", 1)[0]
        if path.endswith((".onnx_data", ".onnx_data_1", ".wasm")):
            return
        super().log_message(fmt, *args)


def _bind_console_close(httpd):
    """Exit as soon as the console window is closed (start.bat)."""
    CTRL_C_EVENT = 0
    CTRL_BREAK_EVENT = 1
    CTRL_CLOSE_EVENT = 2
    CTRL_LOGOFF_EVENT = 5
    CTRL_SHUTDOWN_EVENT = 6
    stop_on = {CTRL_C_EVENT, CTRL_BREAK_EVENT, CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT}

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
    def handler(ctrl_type):
        if ctrl_type in stop_on:
            try:
                httpd.shutdown()
            except Exception:
                pass
            os._exit(0)
        return False

    # Keep a reference: a collected callback makes later closes crash.
    _bind_console_close.handler = handler
    ctypes.windll.kernel32.SetConsoleCtrlHandler(handler, True)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    gemma = os.path.join("models", "onnx-community", "gemma-4-E4B-it-ONNX", "onnx")
    gemma_needed = [
        os.path.join(gemma, "decoder_model_merged_q4f16.onnx"),
        os.path.join(gemma, "decoder_model_merged_q4f16.onnx_data"),
        os.path.join(gemma, "embed_tokens_q4f16.onnx_data"),
        os.path.join("vendor4", "transformers.min.js"),
        os.path.join("vendor4", "ort-wasm-simd-threaded.asyncify.wasm"),
        "worker-v4.js",
    ]
    gemma_missing = [p for p in gemma_needed if not os.path.isfile(p)]
    if gemma_missing:
        print("Gemma 4 E4B is incomplete (run download_gemma4.ps1):")
        for p in gemma_missing:
            print("  -", p)
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    _bind_console_close(httpd)
    print(f"Watermarking playground: http://127.0.0.1:{PORT}/")
    print("Model: Gemma 4 E4B. Needs Chrome or Edge with WebGPU.")
    print("Close the start.bat window to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
