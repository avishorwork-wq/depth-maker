import { AutoModelForDepthEstimation, Tensor, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6';
let Muxer, ArrayBufferTarget;

const $ = id => document.getElementById(id);
const statusEl = $('status'), bar = $('bar'), barfill = $('barfill');
const say = t => { statusEl.textContent = t; };
const sleep = ms => new Promise(r => setTimeout(r, ms));

// ---------- offline + threads: service worker ----------
async function setupSW() {
  if (!('serviceWorker' in navigator)) return;
  try {
    const reg = navigator.serviceWorker.register('2-sw.js');
    if (!navigator.serviceWorker.controller) {
      await Promise.race([reg, sleep(4000)]);
      await Promise.race([navigator.serviceWorker.ready, sleep(6000)]);
    }
    if (!navigator.serviceWorker.controller) {
      // first visit: reload once so every file goes through the cache
      if (!sessionStorage.getItem('dm-reload')) { sessionStorage.setItem('dm-reload', '1'); location.reload(); await new Promise(() => {}); }
    }
    navigator.storage?.persist?.();
  } catch (e) { /* app still works without it, just not offline */ }
}
await setupSW();
try { ({ Muxer, ArrayBufferTarget } = await import('https://cdn.jsdelivr.net/npm/mp4-muxer@5.2.2/+esm')); } catch (e) {}

env.useBrowserCache = false; // the service worker does the caching
env.backends.onnx.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;

// ---------- colour ramps ----------
const RAMPS = {
  Turbo: [[48,18,59],[70,107,227],[40,187,236],[49,241,152],[164,252,60],[237,208,58],[251,128,34],[200,41,3],[122,4,3]],
  Inferno: [[0,0,4],[40,11,84],[101,21,110],[159,42,99],[212,72,66],[245,125,21],[250,193,39],[252,255,164]],
  Gray: [[0,0,0],[255,255,255]],
  Ice: [[5,10,40],[30,70,160],[80,170,230],[200,240,255],[255,255,255]],
};
function lut(stops) {
  const out = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255 * (stops.length - 1), k = Math.min(Math.floor(t), stops.length - 2), f = t - k;
    for (let c = 0; c < 3; c++) out[i * 3 + c] = stops[k][c] + (stops[k + 1][c] - stops[k][c]) * f;
  }
  return out;
}
const LUTS = Object.fromEntries(Object.entries(RAMPS).map(([k, v]) => [k, lut(v)]));

// ---------- effect settings / UI ----------
const state = { fx: 'Turbo', fog: 0.7, fogColor: '#aab4c8', focus: 0.75, blur: 7, strength: 0.06, invert: false, d3: 1.0, points: false };
const EFFECTS = ['Turbo', 'Inferno', 'Gray', 'Ice', 'Fog', 'Focus blur', '3D parallax', '3D view'];
let dirty = true;
const fxBox = $('fx');
for (const name of EFFECTS) {
  const b = document.createElement('button'); b.textContent = name;
  b.onclick = () => { state.fx = name; refreshFx(); dirty = true; };
  fxBox.append(b);
}
const slider = (label, key, min, max, step) => `<label>${label}<input type="range" min="${min}" max="${max}" step="${step}" value="${state[key]}" data-k="${key}"></label>`;
function refreshFx() {
  [...fxBox.children].forEach(b => b.classList.toggle('on', b.textContent === state.fx));
  $('fxtag').textContent = state.fx;
  let h = '';
  if (LUTS[state.fx]) h += `<label><input type="checkbox" data-k="invert" ${state.invert ? 'checked' : ''}> Flip near/far</label>`;
  if (state.fx === 'Fog') h += slider('Fog amount', 'fog', 0, 1, 0.01) + `<label>Colour<input type="color" data-k="fogColor" value="${state.fogColor}"></label>`;
  if (state.fx === 'Focus blur') h += slider('Blur', 'blur', 2, 16, 1) + `<span class="hint">Tap the picture to choose what stays sharp.</span>`;
  if (state.fx === '3D parallax') h += slider('Depth', 'strength', 0.01, 0.15, 0.005) + `<span class="hint">Move your mouse or finger over the picture.</span>`;
  if (state.fx === '3D view') h += slider('Depth', 'd3', 0.3, 2, 0.05) + `<label><input type="checkbox" data-k="points" ${state.points ? 'checked' : ''}> Points</label><span class="hint">Drag to turn it. Scroll or pinch to zoom.</span>`;
  $('opts').innerHTML = h;
  glc.style.display = state.fx === '3D view' ? 'block' : 'none';
  $('opts').querySelectorAll('[data-k]').forEach(el => el.oninput = () => {
    const k = el.dataset.k;
    state[k] = el.type === 'checkbox' ? el.checked : el.type === 'color' ? el.value : parseFloat(el.value);
    dirty = true;
  });
}
// refreshFx moved

// ---------- sizes ----------
const orig = $('orig'), out = $('out');
const octx = orig.getContext('2d', { willReadFrequently: true });
const xctx = out.getContext('2d', { willReadFrequently: true });
const half = document.createElement('canvas'), hctx = half.getContext('2d', { willReadFrequently: true });
const blurC = document.createElement('canvas'), bctx = blurC.getContext('2d', { willReadFrequently: true });
const pre = document.createElement('canvas'), pctx = pre.getContext('2d', { willReadFrequently: true });
let DW = 0, DH = 0;            // display / export size
let PW = 0, PH = 0;            // model input size
const pointer = { x: .5, y: .5, active: false };

const QUALITY = [
  { side: 252, afps: 8, key: 'small' },
  { side: 336, afps: 10, key: 'small' },
  { side: 518, afps: 6, key: 'base' },
];
const gpuOK = !!navigator.gpu && await navigator.gpu.requestAdapter().then(a => !!a).catch(() => false);
let camClamp = false;
const qi = () => { let i = parseInt($('quality').value); if (camClamp && !gpuOK) i = Math.min(i, 1); return i; };
function setSizes(vw, vh) {
  const q = QUALITY[qi()];
  const s = q.side / Math.max(vw, vh);
  PW = Math.max(14, Math.round(vw * s / 14) * 14);
  PH = Math.max(14, Math.round(vh * s / 14) * 14);
  const k = Math.min(1, 720 / Math.max(vw, vh));
  DW = Math.max(2, Math.round(vw * k / 2) * 2); DH = Math.max(2, Math.round(vh * k / 2) * 2);
  orig.width = out.width = blurC.width = DW; orig.height = out.height = blurC.height = DH;
  half.width = DW >> 1; half.height = DH >> 1;
  pre.width = PW; pre.height = PH;
  buildGrid();
  scratch = null;
}

// ---------- bilinear grid (low-res depth -> display res) ----------
let gx0, gx1, gxf, gy0, gy1, gyf;
function buildGrid(tw = DW, th = DH, sw = PW, sh = PH) {
  gx0 = new Int32Array(tw); gx1 = new Int32Array(tw); gxf = new Float32Array(tw);
  gy0 = new Int32Array(th); gy1 = new Int32Array(th); gyf = new Float32Array(th);
  for (let x = 0; x < tw; x++) { const f = Math.max(0, Math.min(sw - 1, (x + .5) * sw / tw - .5)); gx0[x] = f | 0; gx1[x] = Math.min(sw - 1, (f | 0) + 1); gxf[x] = f - (f | 0); }
  for (let y = 0; y < th; y++) { const f = Math.max(0, Math.min(sh - 1, (y + .5) * sh / th - .5)); gy0[y] = f | 0; gy1[y] = Math.min(sh - 1, (f | 0) + 1); gyf[y] = f - (f | 0); }
}
// sample two depth frames (a, b, blend t) into dst (Float32, size tw*th, 0..1)
function upsample(a, b, t, dst, tw, th, sw) {
  const hasB = b && t > 0.001;
  for (let y = 0; y < th; y++) {
    const ra = gy0[y] * sw, rb = gy1[y] * sw, fy = gyf[y];
    for (let x = 0; x < tw; x++) {
      const x0 = gx0[x], x1 = gx1[x], fx = gxf[x];
      let v = (a[ra + x0] * (1 - fx) + a[ra + x1] * fx) * (1 - fy) + (a[rb + x0] * (1 - fx) + a[rb + x1] * fx) * fy;
      if (hasB) {
        const w = (b[ra + x0] * (1 - fx) + b[ra + x1] * fx) * (1 - fy) + (b[rb + x0] * (1 - fx) + b[rb + x1] * fx) * fy;
        v = v * (1 - t) + w * t;
      }
      dst[y * tw + x] = v * (1 / 255);
    }
  }
}

// ---------- edge-aware refinement (fast guided filter, guided by the video frame) ----------
function box(src, dst, tmp, w, h, r) {
  const P = new Float64Array(Math.max(w, h) + 1);
  for (let y = 0; y < h; y++) {
    P[0] = 0; for (let x = 0; x < w; x++) P[x + 1] = P[x] + src[y * w + x];
    for (let x = 0; x < w; x++) { const lo = Math.max(0, x - r), hi = Math.min(w - 1, x + r); tmp[y * w + x] = (P[hi + 1] - P[lo]) / (hi - lo + 1); }
  }
  for (let x = 0; x < w; x++) {
    P[0] = 0; for (let y = 0; y < h; y++) P[y + 1] = P[y] + tmp[y * w + x];
    for (let y = 0; y < h; y++) { const lo = Math.max(0, y - r), hi = Math.min(h - 1, y + r); dst[y * w + x] = (P[hi + 1] - P[lo]) / (hi - lo + 1); }
  }
}
let scratch = null;
function getScratch() {
  if (scratch) return scratch;
  const hw = DW >> 1, hh = DH >> 1, n = hw * hh, f = k => new Float32Array(k);
  scratch = { hw, hh, I: f(n), p: f(n), Ip: f(n), II: f(n), mI: f(n), mp: f(n), mIp: f(n), mII: f(n), a: f(n), b: f(n), ma: f(n), mb: f(n), tmp: f(n), pHalf: f(n) };
  return scratch;
}
let useGuided = true, GEPS = 0.0008, GR = 2;
function refine(depth, srcFull) {
  const S = getScratch(), { hw, hh } = S, n = hw * hh;
  hctx.drawImage(orig, 0, 0, hw, hh);
  const hd = hctx.getImageData(0, 0, hw, hh).data;
  for (let i = 0; i < n; i++) S.I[i] = (hd[i * 4] * 0.299 + hd[i * 4 + 1] * 0.587 + hd[i * 4 + 2] * 0.114) / 255;
  // depth down to half res (average 2x2)
  for (let y = 0; y < hh; y++) for (let x = 0; x < hw; x++) {
    const i = (y * 2) * DW + x * 2;
    S.p[y * hw + x] = (depth[i] + depth[i + 1] + depth[i + DW] + depth[i + DW + 1]) * 0.25;
  }
  for (let i = 0; i < n; i++) { S.Ip[i] = S.I[i] * S.p[i]; S.II[i] = S.I[i] * S.I[i]; }
  const r = GR;
  box(S.I, S.mI, S.tmp, hw, hh, r); box(S.p, S.mp, S.tmp, hw, hh, r); box(S.Ip, S.mIp, S.tmp, hw, hh, r); box(S.II, S.mII, S.tmp, hw, hh, r);
  const eps = GEPS;
  for (let i = 0; i < n; i++) {
    const v = S.mII[i] - S.mI[i] * S.mI[i], c = S.mIp[i] - S.mI[i] * S.mp[i];
    const a = c / (v + eps); S.a[i] = a; S.b[i] = S.mp[i] - a * S.mI[i];
  }
  box(S.a, S.ma, S.tmp, hw, hh, r); box(S.b, S.mb, S.tmp, hw, hh, r);
  // apply at full res: q = a*I + b, with a,b bilinear from half res
  for (let y = 0; y < DH; y++) {
    const fy = Math.max(0, Math.min(hh - 1, (y + .5) / 2 - .5)), y0 = fy | 0, y1 = Math.min(hh - 1, y0 + 1), wy = fy - y0;
    for (let x = 0; x < DW; x++) {
      const fx = Math.max(0, Math.min(hw - 1, (x + .5) / 2 - .5)), x0 = fx | 0, x1 = Math.min(hw - 1, x0 + 1), wx = fx - x0;
      const i00 = y0 * hw + x0, i01 = y0 * hw + x1, i10 = y1 * hw + x0, i11 = y1 * hw + x1;
      const a = (S.ma[i00] * (1 - wx) + S.ma[i01] * wx) * (1 - wy) + (S.ma[i10] * (1 - wx) + S.ma[i11] * wx) * wy;
      const b = (S.mb[i00] * (1 - wx) + S.mb[i01] * wx) * (1 - wy) + (S.mb[i10] * (1 - wx) + S.mb[i11] * wx) * wy;
      const j = (y * DW + x) * 4;
      const I = (srcFull[j] * 0.299 + srcFull[j + 1] * 0.587 + srcFull[j + 2] * 0.114) / 255;
      const q = a * I + b, o = y * DW + x;
      depth[o] = q < 0 ? 0 : q > 1 ? 1 : q;
    }
  }
}

// ---------- real 3D view (WebGL2): the picture is lifted into a surface or point cloud ----------
const glc = document.createElement('canvas');
glc.style.cssText = 'position:absolute;left:0;top:0;width:100%;height:100%;display:none;touch-action:none;cursor:grab';
out.parentElement.append(glc);
const gl = glc.getContext('webgl2', { preserveDrawingBuffer: true, antialias: true });
refreshFx();
const view = { yaw: 0, pitch: 0.05, zoom: 1.0, last: 0 };
let glp = null, glTexD, glTexT, glVao, glIdx = 0, glPts = 0, glGrid = '';
function glInit() {
  if (!gl || glp) return;
  const vs = `#version 300 es
  in vec2 a; uniform sampler2D uD; uniform float uYaw, uPitch, uZoom, uAsp, uScale, uPt;
  out vec2 vUv; out float vD;
  void main(){ vUv = a; float d = textureLod(uD, a, 0.0).r; vD = d;
    vec3 p = vec3((a.x-.5)*2.0*uAsp, (.5-a.y)*2.0, (d-.5)*uScale);
    float cy=cos(uYaw), sy=sin(uYaw), cp=cos(uPitch), sp=sin(uPitch);
    vec3 q = vec3(cy*p.x+sy*p.z, p.y, -sy*p.x+cy*p.z);
    q = vec3(q.x, cp*q.y - sp*q.z, sp*q.y + cp*q.z);
    float k = 1.0/(1.0 - q.z*0.28);
    gl_Position = vec4(q.x*k*uZoom/uAsp, q.y*k*uZoom, -q.z*0.4, 1.0);
    gl_PointSize = uPt * k * uZoom; }`;
  const fs = `#version 300 es
  precision highp float; in vec2 vUv; in float vD; uniform sampler2D uT; uniform float uPt; out vec4 o;
  void main(){ if(uPt < 0.5 && fwidth(vD) > 0.035) discard; o = vec4(texture(uT, vUv).rgb, 1.0); }`;
  const sh = (t, src) => { const x = gl.createShader(t); gl.shaderSource(x, src); gl.compileShader(x); if (!gl.getShaderParameter(x, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(x)); return x; };
  glp = gl.createProgram(); gl.attachShader(glp, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(glp, sh(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(glp);
  glTexD = gl.createTexture(); glTexT = gl.createTexture();
  for (const t of [glTexD, glTexT]) { gl.bindTexture(gl.TEXTURE_2D, t); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); }
}
function glMesh() {
  const key = DW + 'x' + DH; if (key === glGrid) return; glGrid = key;
  const gx = 288, gy = Math.max(8, Math.round(gx * DH / DW)), v = new Float32Array((gx + 1) * (gy + 1) * 2);
  for (let y = 0, k = 0; y <= gy; y++) for (let x = 0; x <= gx; x++) { v[k++] = x / gx; v[k++] = y / gy; }
  const idx = new Uint32Array(gx * gy * 6); let m = 0;
  for (let y = 0; y < gy; y++) for (let x = 0; x < gx; x++) { const a = y * (gx + 1) + x, b = a + 1, c = a + gx + 1, d = c + 1; idx[m++] = a; idx[m++] = b; idx[m++] = c; idx[m++] = b; idx[m++] = d; idx[m++] = c; }
  glVao = gl.createVertexArray(); gl.bindVertexArray(glVao);
  const vb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, vb); gl.bufferData(gl.ARRAY_BUFFER, v, gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(glp, 'a'); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  const ib = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
  glIdx = idx.length; glPts = v.length / 2;
}
let d8 = null;
function renderGL(time) {
  glInit(); if (!glp) return; glMesh();
  if (glc.width !== DW || glc.height !== DH) { glc.width = DW; glc.height = DH; }
  const n = DW * DH; if (!d8 || d8.length !== n) d8 = new Uint8Array(n);
  for (let i = 0; i < n; i++) d8[i] = depthD[i] * 255 + .5;
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, glTexD); gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, DW, DH, 0, gl.RED, gl.UNSIGNED_BYTE, d8);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, glTexT); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, orig);
  gl.viewport(0, 0, DW, DH); gl.clearColor(.05, .055, .075, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT); gl.enable(gl.DEPTH_TEST);
  gl.useProgram(glp);
  const idle = performance.now() - view.last > 3000;
  const yaw = idle ? Math.sin(time / 1600) * 0.38 : view.yaw, pitch = idle ? 0.05 + Math.sin(time / 2300) * 0.08 : view.pitch;
  gl.uniform1i(gl.getUniformLocation(glp, 'uD'), 0); gl.uniform1i(gl.getUniformLocation(glp, 'uT'), 1);
  gl.uniform1f(gl.getUniformLocation(glp, 'uYaw'), yaw); gl.uniform1f(gl.getUniformLocation(glp, 'uPitch'), pitch);
  gl.uniform1f(gl.getUniformLocation(glp, 'uZoom'), view.zoom * 1.2); gl.uniform1f(gl.getUniformLocation(glp, 'uAsp'), DW / DH);
  gl.uniform1f(gl.getUniformLocation(glp, 'uScale'), state.d3 * 1.1); gl.uniform1f(gl.getUniformLocation(glp, 'uPt'), state.points ? Math.max(2.2, DW / 220) : 0);
  gl.bindVertexArray(glVao);
  if (state.points) gl.drawArrays(gl.POINTS, 0, glPts); else gl.drawElements(gl.TRIANGLES, glIdx, gl.UNSIGNED_INT, 0);
}
{ let drag = null;
  glc.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY, yaw: view.last && performance.now() - view.last < 3000 ? view.yaw : 0, pitch: view.pitch }; glc.setPointerCapture(e.pointerId); glc.style.cursor = 'grabbing'; });
  glc.addEventListener('pointermove', e => { if (!drag) return; view.yaw = drag.yaw + (e.clientX - drag.x) / 200; view.pitch = Math.max(-1, Math.min(1, drag.pitch + (e.clientY - drag.y) / 200)); view.last = performance.now(); dirty = true; });
  glc.addEventListener('pointerup', () => { drag = null; glc.style.cursor = 'grab'; });
  glc.addEventListener('wheel', e => { e.preventDefault(); view.zoom = Math.max(.5, Math.min(3, view.zoom * (e.deltaY < 0 ? 1.08 : .93))); view.last = performance.now(); dirty = true; }, { passive: false });
  let pinch = 0; glc.addEventListener('touchmove', e => { if (e.touches.length === 2) { const d = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY); if (pinch) view.zoom = Math.max(.5, Math.min(3, view.zoom * d / pinch)); pinch = d; view.last = performance.now(); dirty = true; } }, { passive: true });
  glc.addEventListener('touchend', () => { pinch = 0; });
}
// ---------- render ----------
let depthD = null;            // Float32 display-res depth
function renderFx(ctx, srcData, t, px, py) {
  const W = DW, H = DH, img = ctx.createImageData(W, H), o = img.data, N = W * H, depth = depthD;
  const L = LUTS[state.fx];
  if (L) {
    for (let i = 0; i < N; i++) { let d = depth[i]; if (state.invert) d = 1 - d; const v = (d * 255) | 0; o[i * 4] = L[v * 3]; o[i * 4 + 1] = L[v * 3 + 1]; o[i * 4 + 2] = L[v * 3 + 2]; o[i * 4 + 3] = 255; }
  } else if (state.fx === 'Fog') {
    const s = srcData, c = parseInt(state.fogColor.slice(1), 16), fr = c >> 16, fg = (c >> 8) & 255, fb = c & 255;
    for (let i = 0; i < N; i++) { const f = Math.min(1, Math.max(0, (1 - depth[i]) * 1.35 - .05)) * state.fog; const j = i * 4; o[j] = s[j] * (1 - f) + fr * f; o[j + 1] = s[j + 1] * (1 - f) + fg * f; o[j + 2] = s[j + 2] * (1 - f) + fb * f; o[j + 3] = 255; }
  } else if (state.fx === 'Focus blur') {
    bctx.filter = `blur(${state.blur}px)`; bctx.drawImage(orig, 0, 0); bctx.filter = 'none';
    const s = srcData, b = bctx.getImageData(0, 0, W, H).data;
    for (let i = 0; i < N; i++) { const f = Math.min(1, Math.abs(depth[i] - state.focus) * 3.2); const j = i * 4; o[j] = s[j] * (1 - f) + b[j] * f; o[j + 1] = s[j + 1] * (1 - f) + b[j + 1] * f; o[j + 2] = s[j + 2] * (1 - f) + b[j + 2] * f; o[j + 3] = 255; }
  } else if (state.fx === '3D parallax') {
    const s = srcData;
    let ax, ay; if (px == null) { ax = Math.sin(t / 900) * .9; ay = Math.cos(t / 1300) * .45; } else { ax = (px - .5) * 2; ay = (py - .5) * 2; }
    const sx = ax * state.strength * W, sy = ay * state.strength * H;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x, k = depth[i] - .5;
      let xx = Math.round(x + k * sx), yy = Math.round(y + k * sy);
      xx = xx < 0 ? 0 : xx >= W ? W - 1 : xx; yy = yy < 0 ? 0 : yy >= H ? H - 1 : yy;
      const j = i * 4, m = (yy * W + xx) * 4; o[j] = s[m]; o[j + 1] = s[m + 1]; o[j + 2] = s[m + 2]; o[j + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}
// draw frame: source canvas `orig` must already hold the current video frame
function drawResult(a, b, t, time) {
  if (!depthD || depthD.length !== DW * DH) depthD = new Float32Array(DW * DH);
  upsample(a, b, t, depthD, DW, DH, PW);
  const srcData = octx.getImageData(0, 0, DW, DH).data;
  if (useGuided) refine(depthD, srcData);
  if (state.fx === '3D view') renderGL(time ?? performance.now());
  else renderFx(xctx, srcData, time ?? performance.now(), pointer.active ? pointer.x : null, pointer.y);
}
out.addEventListener('pointermove', e => { const r = out.getBoundingClientRect(); pointer.x = (e.clientX - r.left) / r.width; pointer.y = (e.clientY - r.top) / r.height; pointer.active = true; if (state.fx === '3D parallax') dirty = true; });
out.addEventListener('pointerleave', () => { pointer.active = false; });
out.addEventListener('pointerdown', e => {
  if (state.fx !== 'Focus blur' || !depthD) return;
  const r = out.getBoundingClientRect(), x = Math.min(DW - 1, Math.max(0, Math.floor((e.clientX - r.left) / r.width * DW))), y = Math.min(DH - 1, Math.max(0, Math.floor((e.clientY - r.top) / r.height * DH)));
  state.focus = depthD[y * DW + x]; dirty = true;
});

// ---------- model ----------
const models = {}; let model = null, device = gpuOK ? 'webgpu' : 'wasm';
const MODELS = {
  small: { id: 'onnx-community/depth-anything-v2-small', gpu: 'fp16', cpu: 'q8', mb: 30 },
  base: { id: 'onnx-community/depth-anything-v2-base', gpu: 'fp16', cpu: 'q4', mb: gpuOK ? 195 : 100 },
};
async function loadModel(key = QUALITY[qi()].key) {
  if (models[key]) { model = models[key]; return model; }
  const M = MODELS[key];
  bar.style.display = 'block'; barfill.style.width = '0%'; say(`Loading the ${key === 'base' ? 'Best' : 'standard'} depth model (first time only, about ${M.mb} MB)...`);
  const progress = p => { if (p.status === 'progress' && p.total) barfill.style.width = (p.loaded / p.total * 100) + '%'; };
  const attempt = (dev, dt) => AutoModelForDepthEstimation.from_pretrained(M.id, { device: dev, dtype: dt, progress_callback: progress });
  let m;
  try { if (!gpuOK) throw 0; m = await attempt('webgpu', M.gpu); device = 'webgpu'; }
  catch (e) { m = await attempt('wasm', M.cpu); device = 'wasm'; }
  models[key] = m; model = m;
  bar.style.display = 'none';
  return m;
}
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
let inBuf = null;
// run the network on whatever is in `pre`; returns {depth: Float32 raw (bigger = nearer), gray: Uint8}
async function infer() {
  const id = pctx.getImageData(0, 0, PW, PH).data, n = PW * PH;
  if (!inBuf || inBuf.length !== 3 * n) inBuf = new Float32Array(3 * n);
  const gray = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const r = id[i * 4], g = id[i * 4 + 1], b = id[i * 4 + 2];
    inBuf[i] = (r / 255 - MEAN[0]) / STD[0]; inBuf[n + i] = (g / 255 - MEAN[1]) / STD[1]; inBuf[2 * n + i] = (b / 255 - MEAN[2]) / STD[2];
    gray[i] = (r * 299 + g * 587 + b * 114) / 1000;
  }
  const res = await model({ pixel_values: new Tensor('float32', inBuf, [1, 3, PH, PW]) });
  const d = res.predicted_depth.data, raw = new Float32Array(n);
  for (let i = 0; i < n; i++) raw[i] = d[i];
  return { raw, gray };
}
// robust range of a raw frame (2nd..98th percentile)
function range(raw) {
  const step = Math.max(1, (raw.length / 4000) | 0), s = [];
  for (let i = 0; i < raw.length; i += step) s.push(raw[i]);
  s.sort((a, b) => a - b);
  return [s[(s.length * .02) | 0], s[Math.min(s.length - 1, (s.length * .98) | 0)]];
}
function quant(raw, lo, hi, dst) {
  const k = 255 / Math.max(1e-6, hi - lo);
  for (let i = 0; i < raw.length; i++) { const v = (raw[i] - lo) * k; dst[i] = v < 0 ? 0 : v > 255 ? 255 : v; }
  return dst;
}

// ---------- sources ----------
const video = document.createElement('video'); video.playsInline = true; video.muted = true;
let mode = null, running = false, busy = false, stream = null, fileURL = null, token = 0;
let cache = null;     // {frames:[Uint8Array], gray:[Uint8Array], n, afps, dur}
window.__depth = { state, view, gl: !!gl, get cache() { return cache; }, get device() { return device; }, get sizes() { return { DW, DH, PW, PH }; }, timings: {} };

function show() { $('empty1').hidden = $('empty2').hidden = true; orig.hidden = out.hidden = false; }
function stopAll() {
  token++; running = false; busy = false;
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  video.pause(); video.srcObject = null;
  $('stop').disabled = true; $('player').hidden = true; bar.style.display = 'none';
}
$('stop').onclick = () => { stopAll(); say('Stopped. Pick another video or use the camera.'); };

// --- upload: analyse once, cache, play ---
async function startFile() {
  stopAll(); mode = 'file'; const my = token; cache = null; camClamp = false;
  await loadModel(); if (my !== token) return;
  video.srcObject = null; video.src = fileURL; video.loop = false;
  await new Promise(r => { video.onloadedmetadata = r; video.load(); });
  const dur = video.duration; if (!isFinite(dur) || dur <= 0) { say('Could not read that video.'); return; }
  setSizes(video.videoWidth, video.videoHeight); show();
  const q = QUALITY[qi()];
  const afps = Math.min(gpuOK ? q.afps * 1.5 : q.afps, 900 / dur), n = Math.max(2, Math.floor(dur * afps));
  const frames = [], grays = [], raws = [], los = [], his = [];
  $('stop').disabled = false; $('dlpng').disabled = true; $('dlvid').disabled = true;
  bar.style.display = 'block'; barfill.style.width = '0%';
  const t0 = performance.now();
  for (let i = 0; i < n; i++) {
    if (my !== token) return;
    const t = Math.min(i / afps, dur - 0.02);
    await new Promise(res => { const f = () => { video.removeEventListener('seeked', f); res(); }; video.addEventListener('seeked', f); video.currentTime = t; });
    pctx.drawImage(video, 0, 0, PW, PH); octx.drawImage(video, 0, 0, DW, DH);
    const ti = performance.now();
    const { raw, gray } = await infer();
    window.__depth.timings.lastInferMs = performance.now() - ti;
    const [lo, hi] = range(raw); raws.push(raw); grays.push(gray); los.push(lo); his.push(hi);
    // live preview of the frame just analysed
    const prev = quant(raw, lo, hi, new Uint8Array(raw.length)); drawResult(prev, null, 0, i * 100);
    const done = i + 1, el = (performance.now() - t0) / 1000, left = el / done * (n - done);
    barfill.style.width = (done / n * 100) + '%';
    say(`Analysing the video once: frame ${done} of ${n}. About ${left < 90 ? Math.ceil(left) + ' sec' : Math.ceil(left / 60) + ' min'} left. (${device === 'webgpu' ? 'GPU' : 'CPU'})`);
  }
  // 1) steady brightness range: smooth lo/hi over time (forward + backward)
  const sm = a => { const f = a.slice(), b = a.slice(); for (let i = 1; i < a.length; i++) f[i] = f[i - 1] * .8 + a[i] * .2; for (let i = a.length - 2; i >= 0; i--) b[i] = b[i + 1] * .8 + a[i] * .2; return a.map((_, i) => (f[i] + b[i]) / 2); };
  const slo = sm(los), shi = sm(his), nPix = PW * PH;
  const q8 = raws.map((r, i) => quant(r, slo[i], shi[i], new Uint8Array(nPix)));
  window.__depth.legacy = window.__keepLegacy ? raws.map(r => { let mn = Infinity, mx = -Infinity; for (let i = 0; i < r.length; i++) { if (r[i] < mn) mn = r[i]; if (r[i] > mx) mx = r[i]; } const u = new Uint8Array(r.length); const k = 255 / (mx - mn); for (let i = 0; i < r.length; i++) u[i] = (r[i] - mn) * k; return u; }) : null;
  // 2) per-pixel temporal smoothing, forward and backward, calmer where the picture is still
  const fwd = [], bwd = new Array(n);
  const blend = (cur, prevS, g, gp, dst) => {
    for (let i = 0; i < nPix; i++) { const m = Math.abs(g[i] - gp[i]) / 255; const a = Math.min(1, .3 + m * 6); dst[i] = prevS[i] * (1 - a) + cur[i] * a; }
  };
  fwd[0] = Float32Array.from(q8[0]);
  for (let i = 1; i < n; i++) { fwd[i] = new Float32Array(nPix); blend(q8[i], fwd[i - 1], grays[i], grays[i - 1], fwd[i]); }
  bwd[n - 1] = fwd[n - 1];
  for (let i = n - 2; i >= 0; i--) { bwd[i] = new Float32Array(nPix); blend(q8[i], bwd[i + 1], grays[i], grays[i + 1], bwd[i]); }
  for (let i = 0; i < n; i++) { const u = new Uint8Array(nPix); for (let k = 0; k < nPix; k++) u[k] = (fwd[i][k] + bwd[i][k]) * .5; frames.push(u); }
  cache = { frames, gray: grays, n, afps, dur, PW, PH, rawNorm: null };
  window.__depth.timings.analysisSec = (performance.now() - t0) / 1000;
  bar.style.display = 'none';
  startPlayback(my);
}
function startPlayback(my) {
  mode = 'play'; running = true; video.loop = true; video.currentTime = 0; video.play().catch(() => {});
  $('player').hidden = false; $('play').textContent = 'Pause'; $('stop').disabled = false;
  $('dlpng').disabled = false; $('dlvid').disabled = !(Muxer && window.VideoEncoder);
  $('vidhint').textContent = Muxer && window.VideoEncoder ? 'Effects and downloads use the saved depth, no waiting.' : 'Video download is not supported in this browser. Image download works.';
  say(`Done. Playing from the saved depth (${cache.n} frames). Switch effects any time.`);
  dirty = true; requestAnimationFrame(playLoop);
}
let lastKey = '';
function playLoop() {
  if (!running || mode !== 'play') return;
  const t = video.currentTime, fi = Math.min(cache.n - 1, t * cache.afps), i0 = Math.floor(fi), i1 = Math.min(cache.n - 1, i0 + 1), f = fi - i0;
  const key = `${i0}:${f.toFixed(2)}:${video.paused}`;
  const par = state.fx === '3D parallax' || state.fx === '3D view';
  if (dirty || key !== lastKey || par) {
    const t0 = performance.now();
    octx.drawImage(video, 0, 0, DW, DH);
    drawResult(cache.frames[i0], cache.frames[i1], f, performance.now());
    window.__depth.timings.renderMs = performance.now() - t0;
    lastKey = key; dirty = false;
    if (!seeking) $('seek').value = Math.round(t / cache.dur * 1000);
    $('time').textContent = `${fmt(t)} / ${fmt(cache.dur)}`;
  }
  requestAnimationFrame(playLoop);
}
const fmt = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
let seeking = false;
$('seek').oninput = e => { seeking = true; if (cache) { video.currentTime = e.target.value / 1000 * cache.dur; dirty = true; } };
$('seek').onchange = () => { seeking = false; };
$('play').onclick = () => { if (video.paused) { video.play(); $('play').textContent = 'Pause'; } else { video.pause(); $('play').textContent = 'Play'; } };

// --- live camera: separate mode ---
async function startCam() {
  stopAll(); mode = 'cam'; const my = token; cache = null; camClamp = true;
  await loadModel(); if (my !== token) return;
  try { stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 } }, audio: false }); }
  catch (e) { say('Camera blocked or not found. Allow camera access, or upload a video instead.'); return; }
  video.srcObject = stream; video.loop = false; await video.play();
  await new Promise(r => { if (video.videoWidth) r(); else video.onloadedmetadata = r; });
  setSizes(video.videoWidth, video.videoHeight); show();
  running = true; $('stop').disabled = false; $('dlpng').disabled = false; $('dlvid').disabled = true;
  $('vidhint').textContent = 'Video download works with an uploaded video. The camera can save images.';
  let prev = null, cur = null, tCur = 0, interval = 250, lo = null, hi = null, grayPrev = null, fps = 0;
  const nPix = PW * PH;
  (async () => {
    while (running && my === token) {
      if (video.readyState < 2) { await sleep(30); continue; }
      const t0 = performance.now();
      pctx.drawImage(video, 0, 0, PW, PH);
      const { raw, gray } = await infer();
      const [l, h] = range(raw); lo = lo == null ? l : lo * .7 + l * .3; hi = hi == null ? h : hi * .7 + h * .3;
      const nu = quant(raw, lo, hi, new Uint8Array(nPix));
      if (cur) for (let i = 0; i < nPix; i++) { const m = Math.abs(gray[i] - grayPrev[i]) / 255; const a = Math.min(1, .35 + m * 6); nu[i] = cur[i] * (1 - a) + nu[i] * a; }
      prev = cur || nu; cur = nu; grayPrev = gray;
      const now = performance.now(); interval = interval * .7 + (now - tCur) * .3; tCur = now;
      const dt = now - t0; fps = fps ? fps * .8 + 1000 / dt * .2 : 1000 / dt;
      window.__depth.timings.camFps = fps; window.__depth.camReady = true;
      say(`Live on ${device === 'webgpu' ? 'GPU' : 'CPU'}: ${fps.toFixed(1)} depth updates per second. The picture itself stays smooth.`);
      await sleep(0);
    }
  })();
  const loop = () => {
    if (!running || mode !== 'cam') return;
    if (cur) {
      octx.drawImage(video, 0, 0, DW, DH);
      const f = Math.min(1, (performance.now() - tCur) / Math.max(60, interval));
      drawResult(prev, cur, f, performance.now());
    } else octx.drawImage(video, 0, 0, DW, DH);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}
$('cam').onclick = startCam;
$('file').onchange = e => { const f = e.target.files[0]; if (!f) return; if (fileURL) URL.revokeObjectURL(fileURL); fileURL = URL.createObjectURL(f); startFile(); e.target.value = ''; };
$('quality').onchange = () => { if (mode === 'file' || mode === 'play') { if (fileURL) startFile(); } else if (mode === 'cam') startCam(); };

// ---------- downloads (reuse the cache, never re-run the model) ----------
function save(blob, name) { const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 4000); }
$('dlpng').onclick = () => (state.fx === '3D view' ? glc : out).toBlob(b => save(b, `depth-${state.fx.replace(/\W+/g, '-').toLowerCase()}.png`), 'image/png');
$('dlvid').onclick = async () => {
  if (mode !== 'play' || !cache || !Muxer) return;
  running = false; video.pause(); $('dlvid').disabled = true; bar.style.display = 'block'; barfill.style.width = '0%';
  const efps = 24, total = Math.max(1, Math.floor(cache.dur * efps));
  const target = new ArrayBufferTarget();
  const muxer = new Muxer({ target, video: { codec: 'avc', width: DW, height: DH }, fastStart: 'in-memory' });
  const enc = new VideoEncoder({ output: (c, m) => muxer.addVideoChunk(c, m), error: e => say('Encode error: ' + e.message) });
  enc.configure({ codec: 'avc1.4D0028', width: DW, height: DH, bitrate: 6_000_000, framerate: efps });
  try {
    for (let k = 0; k < total; k++) {
      const t = Math.min(k / efps, cache.dur - 0.02), fi = Math.min(cache.n - 1, t * cache.afps), i0 = Math.floor(fi), i1 = Math.min(cache.n - 1, i0 + 1);
      await new Promise(res => { const f = () => { video.removeEventListener('seeked', f); res(); }; video.addEventListener('seeked', f); video.currentTime = t; });
      octx.drawImage(video, 0, 0, DW, DH);
      drawResult(cache.frames[i0], cache.frames[i1], fi - i0, t * 1000);
      const fr = new VideoFrame(state.fx === '3D view' ? glc : out, { timestamp: Math.round(k * 1e6 / efps), duration: Math.round(1e6 / efps) });
      enc.encode(fr, { keyFrame: k % 48 === 0 }); fr.close();
      barfill.style.width = ((k + 1) / total * 100) + '%'; say(`Saving video: frame ${k + 1} of ${total}`);
      if (enc.encodeQueueSize > 6) await enc.flush();
    }
    await enc.flush(); muxer.finalize();
    save(new Blob([target.buffer], { type: 'video/mp4' }), `depth-${state.fx.replace(/\W+/g, '-').toLowerCase()}.mp4`);
    say('Done. Your video was saved.');
  } catch (e) { say('Could not save the video: ' + (e.message || e)); }
  bar.style.display = 'none'; $('dlvid').disabled = false; running = true; video.play().catch(() => {}); dirty = true; requestAnimationFrame(playLoop);
};

// ---------- start ----------
window.__depth.setGuided = v => { useGuided = v; dirty = true; };
window.__depth.seek = t => { video.pause(); video.currentTime = t; dirty = true; };
window.__depth.setG = (e, r) => { GEPS = e; GR = r; dirty = true; };
loadModel('small').then(() => say('Ready. Upload a video or use the live camera.')).catch(e => say('Could not load the model: ' + (e.message || e)));
