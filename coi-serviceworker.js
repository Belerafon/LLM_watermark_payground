/* Cross-origin isolation for hosts that cannot set COOP/COEP (GitHub Pages).
   If the server already sent those headers, this does nothing. */
if (typeof window === "undefined") {
  self.addEventListener("install", () => self.skipWaiting());
  self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));
  self.addEventListener("fetch", (event) => {
    if (event.request.cache === "only-if-cached" && event.request.mode !== "same-origin") return;
    const url = new URL(event.request.url);
    if (url.origin !== self.location.origin) return;
    event.respondWith(
      fetch(event.request).then((response) => {
        if (response.status === 0) return response;
        const headers = new Headers(response.headers);
        headers.set("Cross-Origin-Embedder-Policy", "credentialless");
        headers.set("Cross-Origin-Opener-Policy", "same-origin");
        return new Response(response.body, {
          status: response.status,
          statusText: response.statusText,
          headers,
        });
      })
    );
  });
} else if (window.crossOriginIsolated === false && window.navigator?.serviceWorker) {
  const src = document.currentScript?.src;
  if (src) {
    window.navigator.serviceWorker.register(src).then((registration) => {
      if (registration.active && !window.navigator.serviceWorker.controller) {
        window.location.reload();
      }
    });
  }
}
