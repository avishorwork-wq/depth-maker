// Depth Maker service worker: offline cache + headers for multi-threaded wasm.
const SHELL = 'depthmaker-shell-v1';
const LIBS = 'depthmaker-libs-v1';
const SHELL_FILES = ['./', './1-app.js', './3-manifest.webmanifest', './4-icon192.png', './5-icon512.png'];
self.addEventListener('install', e => {
  e.waitUntil(caches.open(SHELL).then(c => c.addAll(SHELL_FILES.map(u => new Request(u, {cache: 'reload'})))).then(() => self.skipWaiting()));
});
self.addEventListener('activate', e => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) if (k !== SHELL && k !== LIBS) await caches.delete(k);
    await self.clients.claim();
  })());
});
function iso(res) {
  if (!res || res.status === 0 || res.type === 'opaque') return res;
  const h = new Headers(res.headers);
  h.set('Cross-Origin-Opener-Policy', 'same-origin');
  h.set('Cross-Origin-Embedder-Policy', 'credentialless');
  return new Response(res.body, {status: res.status, statusText: res.statusText, headers: h});
}
self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET' || req.headers.has('range')) return;
  const url = new URL(req.url);
  if (url.origin === location.origin) {
    // same site: show the cached copy now, refresh it in the background
    e.respondWith((async () => {
      const cache = await caches.open(SHELL);
      const hit = await cache.match(req, {ignoreSearch: req.mode === 'navigate'});
      const net = fetch(req).then(r => { if (r.ok) cache.put(req, r.clone()); return r; }).catch(() => null);
      if (hit) { e.waitUntil(net); return iso(hit); }
      const r = await net;
      return r ? iso(r) : new Response('Offline and not cached yet.', {status: 503});
    })());
  } else {
    // libraries and model files: cached forever after the first download
    e.respondWith((async () => {
      const cache = await caches.open(LIBS);
      const hit = await cache.match(req);
      if (hit) return iso(hit);
      const r = await fetch(req);
      if (r.ok && r.type !== 'opaque') cache.put(req, r.clone());
      return iso(r);
    })());
  }
});
