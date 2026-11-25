/**
 * Morph core for the 2.5D sprite avatar: landmarks + shared triangles + lerp + draw.
 *
 * Framework-free and DOM-light on purpose, so it ports to React Native (react-native-skia
 * `drawVertices` takes exactly these arrays: positions, texture coordinates, triangle indices):
 *   - MorphTrack: one region's state machine. A transition A -> B moves the mesh from the CURRENT
 *     geometry to B's landmarks with an ease-in-out and dissolves the textures; retargeting mid-way
 *     starts from where the mesh is (no geometric pop; acts as a low-pass on the jaw).
 *   - GLMeshRenderer: WebGL1, precomputed UV buffers; drawAdaptive() = one call per region (below).
 *   - drawMesh2D: CPU canvas fallback (per-triangle affine + clip).
 * Data comes from scripts/avatar3d/sprite/morph_data.py (morph.json): per region a box, a triangle
 * list shared by all states, and per state a texture (same alpha for all states) + points.
 */

export const easeInOut = (t) => (t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t));
const clamp01 = (x) => Math.min(1, Math.max(0, x));
const MIN_WEIGHT = 0.002;
// The textures do NOT dissolve in step with the geometry: a linear dissolve showed both images at 50%
// half-way (the iris through a see-through lid mid-blink, two lip outlines mid-viseme). The mesh warps
// the whole way; the texture switch happens only inside this centred window of the transition, where
// both warped images already line up.
// With one-skin textures (morph_data v4+) the skin is the same photo in every state; only real features differ, and a
// longer fade (0.5 was tried) shows two expressions at once (doubled lids / teeth): they switch in this short window.
export const SWITCH_WINDOW = 0.12;
export const switchWeight = (u, win = SWITCH_WINDOW) => easeInOut(clamp01((u - (1 - win) / 2) / win));
const MAX_DRAWS = 3;
// Adaptive dissolve (drawAdaptive): the fast switch above is right for FEATURES (lids, teeth, tongue: a slow
// dissolve shows them see-through), but on skin it shows as a jump of the whole texture at the switch moment (the
// stubble / tone of the two states never line up exactly: "the image gets replaced"). So every track also keeps
// WIDE weights that dissolve in step with the geometry, and the shader picks per pixel: where the wide and the
// narrow mixture look the same at a blurred scale (skin, stubble, tone) it shows the wide one (smooth); where they
// differ in structure (a feature appears / disappears) it shows the narrow one (no ghost).
// Stretch: a state's texture magnified a lot at a pixel (closed lips pulled over an open mouth: vertical streaks)
// shows nothing real there, so at that pixel its weight goes to the other states (the open mouth's interior).
// density = texels per screen pixel (sqrt of the warp's area ratio): below stretchLo no weight, above stretchHi full.
// Needs the constrained opening mesh (morph_data v5): with triangles zig-zagging across the lips it could not help.
export const ADAPTIVE = { radiusTexels: 4, lo: 0.05, hi: 0.12, stretchLo: 0.35, stretchHi: 0.65 };

const mixMap = (pairs, scale, into = new Map()) => {
  for (const [k, v] of pairs) into.set(k, (into.get(k) ?? 0) + v * scale);
  return into;
};

// Motion: a retarget starts from the current position AND the current velocity (cubic Hermite), so speech flows from
// shape to shape instead of stopping at every viseme (ease-in-out per segment stopped the mouth 10-20 times a
// second). The carried velocity is capped per point at the remaining distance (no overshoot into folds).
const hermite = (p0, v0, p1, dur, s, out) => {
  const s2 = s * s, s3 = s2 * s;
  const h00 = 2 * s3 - 3 * s2 + 1, h10 = (s3 - 2 * s2 + s) * dur, h01 = -2 * s3 + 3 * s2;
  for (let i = 0; i < p0.length; i++) out[i] = h00 * p0[i] + h10 * v0[i] + h01 * p1[i];
  return out;
};
const hermiteVel = (p0, v0, p1, dur, s, out) => {  // d/dt, px per ms
  const d00 = (6 * s * s - 6 * s) / dur, d10 = 3 * s * s - 4 * s + 1, d01 = (-6 * s * s + 6 * s) / dur;
  for (let i = 0; i < p0.length; i++) out[i] = d00 * p0[i] + d10 * v0[i] + d01 * p1[i];
  return out;
};

// Physical smoothing: the drawn mesh can follow the target path through a critically damped spring (no bounce) with
// this time constant. OFF: measured with morph_render.py (sync_lag_ms / sync_reach / pp_gap_iod), a 25 ms spring made
// the lips lag the audio by 17-33 ms, reach only 81-85% of each shape and stop closing for p/b/m - the words no longer
// matched the lips. The Hermite path above already keeps the motion continuous from shape to shape without any lag.
// A jump in time (> SPRING_RESET_MS, a paused tab or a QA render) snaps to the path.
export const SPRING_MS = 0;
const SPRING_RESET_MS = 250;
const SPRING_STEP_MS = 4;

// --- speech timeline (zero-lag smoothing) -------------------------------------------------------------------------
// During speech the whole sequence of mouth shapes is known in advance, so the mesh is the shapes' step function
// smoothed by a Gaussian CENTRED on the current time: w_k(t) = Phi((t - start_k) / s) - Phi((t - end_k) / s). No lag
// (a causal spring lagged the audio by 17-33 ms), no stop at every shape (per-segment easing stopped the mouth 10-20
// times a second), and short shapes blend with their neighbours the way real lips do. At a boundary the smaller of
// the two shapes' sigmas is used, so the weights always sum to 1; shapes whose lips touch (contact) get the small
// sigma, so p / b / m still close fully.
export const TIMELINE_SIGMA_S = 0.022;
export const TIMELINE_CONTACT_SIGMA_S = 0.008;
const TIMELINE_SPAN = 4;        // shapes further than this x sigma from the current time have no weight
export const TIMELINE_SHARPEN = 8;     // narrow (feature) weights = w^this, normalised: features switch near a crossover

const erf = (x) => {  // Abramowitz-Stegun 7.1.26, |error| < 1.5e-7
  const s = Math.sign(x), a = Math.abs(x), t = 1 / (1 + 0.3275911 * a);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
    * Math.exp(-a * a);
  return s * y;
};
const phi = (x) => 0.5 * (1 + erf(x / Math.SQRT2));

/** Projection of `cur` on the segment from -> to (0..1); `fallback` when from and to are the same shape. */
function geoProgress(from, to, cur, fallback) {
  let num = 0, den = 0;
  for (let i = 0; i < from.length; i++) {
    const d = to[i] - from[i];
    num += (cur[i] - from[i]) * d;
    den += d * d;
  }
  return den > 1 ? clamp01(num / den) : fallback;
}

export function lerpPoints(a, b, t, out = new Float32Array(a.length)) {
  for (let i = 0; i < a.length; i++) out[i] = a[i] + (b[i] - a[i]) * t;
  return out;
}

/** One region: current geometry + texture weights, retargetable at any time. */
export class MorphTrack {
  /** @param {Object<string, Float32Array>} points state id -> flat [x0,y0,x1,y1,...] (image px) */
  constructor(points, initial = "base") {
    this.points = points;
    this.rest = initial;
    this.to = initial;
    this.amount = 1;
    this.toPoints = points[initial];
    this.from = { points: points[initial].slice(), mix: [[initial, 1]], wide: [[initial, 1]] };
    this.vel = new Float32Array(points[initial].length);
    this.start = 0;
    this.dur = 1;
    this.cur = points[initial].slice();
  }

  /**
   * Move towards `target` over `durMs` (eased), starting from the geometry shown at `now`. `amount` < 1
   * stops the geometry part of the way from the rest state to the target (the target's texture is still
   * the one shown): speech uses this, since image-model viseme keyframes are exaggerated poses.
   */
  set(target, now, durMs, amount = 1) {
    if (this.timeline) return;  // speech timeline in charge (setTimeline)
    if (!(target in this.points) || (target === this.to && amount === this.amount)) return;
    const snap = this.sample(now);
    const path = this.cur.slice();  // the next segment starts on the target path (the spring smooths what is drawn)
    const s = clamp01((now - this.start) / this.dur);
    const v = s < 1 ? hermiteVel(this.from.points, this.vel, this.toPoints, this.dur, s, new Float32Array(this.vel.length))
      : new Float32Array(this.vel.length);
    this.amount = amount;
    this.toPoints = amount === 1 ? this.points[target]
      : lerpPoints(this.points[this.rest], this.points[target], amount);
    this.from = { points: path, mix: snap.draws.map(([k, w]) => [k, w]),
      wide: snap.layers.map(([k, , w]) => [k, w]).filter(([, w]) => w > MIN_WEIGHT) };
    this.to = target;
    this.start = now;
    this.dur = Math.max(1, durMs);
    for (let i = 0; i < v.length; i += 2) {  // cap the carried displacement at the remaining distance
      const rx = this.toPoints[i] - path[i], ry = this.toPoints[i + 1] - path[i + 1];
      const cx = v[i] * this.dur, cy = v[i + 1] * this.dur;
      const k = Math.min(1, Math.hypot(rx, ry) / (Math.hypot(cx, cy) || 1));
      v[i] *= k; v[i + 1] *= k;
    }
    this.vel = v;
  }

  /**
   * {points (Float32Array, image px), draws: [[stateId, weight], ...] (narrow switch, summing to 1, largest
   * first: the 2D fallback), layers: [[stateId, narrowWeight, wideWeight], ...] (at most MAX_DRAWS, each
   * column summing to 1: drawAdaptive)}
   */
  sample(now) {
    if (this.timeline) return this.sampleTimeline();
    const s0 = clamp01((now - this.start) / this.dur);
    hermite(this.from.points, this.vel, this.toPoints, this.dur, s0, this.cur);
    const points = this.spring(now);
    // texture progress follows the DRAWN shape (where the mesh is between the start and target shapes), not the
    // clock: with the spring the shape lags the clock, and a texture that arrives first is stretched over it
    const u = SPRING_MS ? geoProgress(this.from.points, this.toPoints, points, easeInOut(s0)) : easeInOut(s0);
    const s = switchWeight(u, this.switchWindow ?? SWITCH_WINDOW);
    const n = mixMap([[this.to, 1]], s, mixMap(this.from.mix, 1 - s));
    const wd = mixMap([[this.to, 1]], u, mixMap(this.from.wide, 1 - u));
    let draws = [...n.entries()].filter(([, v]) => v > MIN_WEIGHT).sort((a, b) => b[1] - a[1]).slice(0, MAX_DRAWS);
    const sum = draws.reduce((t, [, v]) => t + v, 0) || 1;
    draws = draws.map(([k, v]) => [k, v / sum]);
    const keys = [...new Set([...n.keys(), ...wd.keys()])]
      .map((k) => [k, n.get(k) ?? 0, wd.get(k) ?? 0])
      .filter(([, a, b]) => Math.max(a, b) > MIN_WEIGHT)
      .sort((a, b) => Math.max(b[1], b[2]) - Math.max(a[1], a[2])).slice(0, MAX_DRAWS);
    const sn = keys.reduce((t, k) => t + k[1], 0) || 1, sw = keys.reduce((t, k) => t + k[2], 0) || 1;
    const layers = keys.map(([k, a, b]) => [k, a / sn, b / sw]);
    return { points, draws, layers };
  }

  /** The spring-smoothed mesh at `now` (see SPRING_MS); this.cur is the target path. */
  spring(now) {
    if (!SPRING_MS) return this.cur;
    const dt = this.springT === undefined ? Infinity : now - this.springT;
    if (!this.sx || dt < 0 || dt > SPRING_RESET_MS) {
      this.sx = this.cur.slice();
      this.sv = new Float32Array(this.cur.length);
      this.springT = now;
      return this.sx;
    }
    const w = 1 / SPRING_MS;
    for (let left = dt; left > 0; left -= SPRING_STEP_MS) {
      const h = Math.min(SPRING_STEP_MS, left);
      for (let i = 0; i < this.sx.length; i++) {
        this.sv[i] += (w * w * (this.cur[i] - this.sx[i]) - 2 * w * this.sv[i]) * h;
        this.sx[i] += this.sv[i] * h;
      }
    }
    this.springT = now;
    return this.sx;
  }

  /**
   * Speech: follow `keys` = [{start, end, state}] (seconds, contiguous, covering the clip) at clock() seconds until
   * clearTimeline(). contact: set of states whose lips touch (small sigma). set() is ignored meanwhile.
   */
  setTimeline(keys, clock, contact = new Set(), opts = {}) {
    const sigma = opts.sigma ?? TIMELINE_SIGMA_S, contactSigma = opts.contactSigma ?? TIMELINE_CONTACT_SIGMA_S;
    // boundary b (between keys b-1 and b): sharp only when the lips CLOSE into contact there (they meet fast);
    // opening and moving between open shapes stay smooth
    const sb = keys.map((k, b) => (b && contact.has(k.state) && !contact.has(keys[b - 1].state)
      ? contactSigma : sigma));
    // undershoot: a sound shorter than opts.undershoot s reaches only (its length / that) of its pose, as in real
    // speech; lips-together (contact) shapes always reach theirs so p / b / m still close. 0 = off.
    const us = opts.undershoot ?? 0;
    this.timeline = { clock, amount: opts.amount ?? 1, rest: opts.rest, sharpen: opts.sharpen ?? TIMELINE_SHARPEN,
      keys: keys.map((k, i) => ({ ...k, sl: sb[i], sr: i < keys.length - 1 ? sb[i + 1] : sb[i],
        reach: us > 0 && !contact.has(k.state) && k.state !== opts.rest ? Math.min(1, (k.end - k.start) / us) : 1 })) };
  }

  /** Back to set(): continue from the shape and textures shown at `now`, towards `target`. */
  clearTimeline(now, target, durMs) {
    if (!this.timeline) return;
    const snap = this.sample(now);
    this.timeline = null;
    this.from = { points: snap.points.slice(), mix: snap.draws.map(([k, w]) => [k, w]),
      wide: snap.layers.map(([k, , w]) => [k, w]).filter(([, w]) => w > MIN_WEIGHT) };
    this.vel = new Float32Array(this.vel.length);
    this.to = target in this.points ? target : this.to;
    this.toPoints = this.points[this.to];
    this.amount = 1;
    this.start = now;
    this.dur = Math.max(1, durMs);
  }

  sampleTimeline() {
    const t = this.timeline.clock();
    const w = new Map();
    const geo = new Map();  // geometry weights: a short sound's share goes partly to the rest mouth (undershoot)
    const rest = this.timeline.rest in this.points ? this.timeline.rest : null;
    for (const k of this.timeline.keys) {
      if (k.end < t - TIMELINE_SPAN * k.sr || k.start > t + TIMELINE_SPAN * k.sl) continue;
      const v = phi((t - k.start) / k.sl) - phi((t - k.end) / k.sr);
      if (v <= MIN_WEIGHT) continue;
      w.set(k.state, (w.get(k.state) ?? 0) + v);
      geo.set(k.state, (geo.get(k.state) ?? 0) + v * k.reach);
      if (rest && k.reach < 1) geo.set(rest, (geo.get(rest) ?? 0) + v * (1 - k.reach));
    }
    if (!w.size) w.set(this.to, 1);
    const sum = [...geo.values()].reduce((a, b) => a + b, 0);
    this.cur.fill(0);
    for (const [s, v] of geo) {
      const p = this.points[s];
      for (let i = 0; i < p.length; i++) this.cur[i] += (v / sum) * p[i];
    }
    const { amount } = this.timeline;
    if (amount !== 1 && rest) {  // shapes played at `amount` of their pose, from the rest mouth
      const r = this.points[rest];
      for (let i = 0; i < r.length; i++) this.cur[i] = r[i] + amount * (this.cur[i] - r[i]);
    }
    const ranked = [...w.entries()].sort((a, b) => b[1] - a[1]).slice(0, MAX_DRAWS);
    const sw = ranked.reduce((a, [, v]) => a + v, 0) || 1;
    const k = this.timeline.sharpen;
    const sh = ranked.reduce((a, [, v]) => a + v ** k, 0) || 1;
    const layers = ranked.map(([s, v]) => [s, v ** k / sh, v / sw]);
    const draws = layers.map(([s, n]) => [s, n]).filter(([, n]) => n > MIN_WEIGHT);
    this.to = ranked[0][0];
    return { points: this.cur, draws, layers };
  }

  get settled() { return this.from.mix.length === 1 && this.from.mix[0][0] === this.to; }
}

/**
 * Sequential "over" alphas that reproduce weighted averages: draw i with alpha w_i / (w_1 + ... + w_i).
 * The first draw is opaque (inside the region mask), so the base's own feature never shows through.
 */
export function overAlphas(draws) {
  let acc = 0;
  return draws.map(([, w]) => { acc += w; return w / acc; });
}

// --- WebGL ---------------------------------------------------------------------------------------
const AVS = `attribute vec2 a_pos; attribute vec2 a_uv0; attribute vec2 a_uv1; attribute vec2 a_uv2; uniform vec2 u_res;
varying vec2 v_uv0; varying vec2 v_uv1; varying vec2 v_uv2;
void main() { vec2 c = a_pos / u_res * 2.0 - 1.0; gl_Position = vec4(c.x, -c.y, 0.0, 1.0);
  v_uv0 = a_uv0; v_uv1 = a_uv1; v_uv2 = a_uv2; }`;
// up to 3 states: N = narrow mixture, W = wide mixture; d = |blur(N) - blur(W)| from a ring of 8 taps + centre
const AFS = `#extension GL_OES_standard_derivatives : enable
precision highp float;
varying vec2 v_uv0; varying vec2 v_uv1; varying vec2 v_uv2;
uniform sampler2D u_t0; uniform sampler2D u_t1; uniform sampler2D u_t2;
uniform vec3 u_wn; uniform vec3 u_ww; uniform vec2 u_texel; uniform float u_r; uniform float u_lo; uniform float u_hi;
uniform float u_slo; uniform float u_shi;
float trust(vec2 uv) {
  vec2 dx = dFdx(uv) / u_texel; vec2 dy = dFdy(uv) / u_texel;
  return smoothstep(u_slo, u_shi, sqrt(abs(dx.x * dy.y - dx.y * dy.x)));
}
// a stretched state's weight moves to the trusted states, in proportion to their presence in either mixture (the
// fast one may already give 0 to the state the mesh still looks like); unchanged if every state is stretched
vec3 reweigh(vec3 w, vec3 t, vec3 q) {
  vec3 r = w * t; float lost = 1.0 - (r.x + r.y + r.z);
  vec3 p = t * q; float sp = p.x + p.y + p.z;
  return sp > 1e-4 ? r + lost * p / sp : w;
}
void main() {
  vec4 c0 = texture2D(u_t0, v_uv0); vec3 c1 = texture2D(u_t1, v_uv1).rgb; vec3 c2 = texture2D(u_t2, v_uv2).rgb;
  vec3 t = vec3(trust(v_uv0), trust(v_uv1), trust(v_uv2));
  vec3 q = u_wn + u_ww;
  vec3 wn = reweigh(u_wn, t, q); vec3 ww = reweigh(u_ww, t, q);
  vec3 dw = wn - ww;
  vec3 bd = dw.x * c0.rgb + dw.y * c1 + dw.z * c2;
  for (int k = 0; k < 8; k++) {
    float a = float(k) * 0.7853982;
    vec2 o = vec2(cos(a), sin(a)) * u_r * u_texel;
    bd += dw.x * texture2D(u_t0, v_uv0 + o).rgb + dw.y * texture2D(u_t1, v_uv1 + o).rgb
        + dw.z * texture2D(u_t2, v_uv2 + o).rgb;
  }
  float d = length(bd / 9.0) * 0.57735;
  float agree = 1.0 - smoothstep(u_lo, u_hi, d);
  vec3 n = wn.x * c0.rgb + wn.y * c1 + wn.z * c2;
  vec3 w = ww.x * c0.rgb + ww.y * c1 + ww.z * c2;
  gl_FragColor = vec4(mix(n, w, agree), c0.a);
}`;
const VS = `attribute vec2 a_pos; attribute vec2 a_uv; uniform vec2 u_res; varying vec2 v_uv;
void main() { vec2 c = a_pos / u_res * 2.0 - 1.0; gl_Position = vec4(c.x, -c.y, 0.0, 1.0); v_uv = a_uv; }`;
const FS = `precision mediump float; varying vec2 v_uv; uniform sampler2D u_tex; uniform float u_alpha;
void main() { vec4 c = texture2D(u_tex, v_uv); gl_FragColor = vec4(c.rgb, c.a * u_alpha); }`;

export class GLMeshRenderer {
  /** @param {HTMLCanvasElement|OffscreenCanvas} canvas */
  constructor(canvas) {
    const gl = canvas.getContext("webgl", { premultipliedAlpha: false, alpha: false, antialias: false });
    if (!gl) throw new Error("no WebGL");
    this.gl = gl;
    this.canvas = canvas;
    const sh = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, sh(gl.VERTEX_SHADER, VS));
    gl.attachShader(p, sh(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(p);
    gl.useProgram(p);
    this.loc = { pos: gl.getAttribLocation(p, "a_pos"), uv: gl.getAttribLocation(p, "a_uv"),
      res: gl.getUniformLocation(p, "u_res"), alpha: gl.getUniformLocation(p, "u_alpha") };
    gl.enableVertexAttribArray(this.loc.pos);
    gl.enableVertexAttribArray(this.loc.uv);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    this.textures = new Map();
    this.texSize = new Map();
    this.meshes = new Map();
    this.posBuf = gl.createBuffer();
    this.prog = p;
    gl.getExtension("OES_standard_derivatives");  // dFdx / dFdy for the stretch measure (WebGL1 extension)
    this.adaptive = this.program(sh, AVS, AFS, ["a_pos", "a_uv0", "a_uv1", "a_uv2"],
      ["u_res", "u_t0", "u_t1", "u_t2", "u_wn", "u_ww", "u_texel", "u_r", "u_lo", "u_hi", "u_slo", "u_shi"]);
    gl.useProgram(p);
  }

  program(sh, vs, fs, attribs, uniforms) {
    const gl = this.gl;
    const p = gl.createProgram();
    gl.attachShader(p, sh(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const loc = { p };
    for (const a of attribs) loc[a] = gl.getAttribLocation(p, a);
    for (const u of uniforms) loc[u] = gl.getUniformLocation(p, u);
    return loc;
  }

  texture(key, img) {
    const gl = this.gl;
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    this.textures.set(key, t);
    this.texSize.set(key, [img.width, img.height]);
  }

  /** A mesh = triangle indices + one UV buffer per state (computed once). */
  mesh(key, triangles, uvByState) {
    const gl = this.gl;
    const idx = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idx);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(triangles.flat()), gl.STATIC_DRAW);
    const uvs = {};
    for (const [s, uv] of Object.entries(uvByState)) {
      const b = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.bufferData(gl.ARRAY_BUFFER, uv, gl.STATIC_DRAW);
      uvs[s] = b;
    }
    this.meshes.set(key, { idx, uvs, count: triangles.length * 3 });
  }

  begin(w, h) {
    const gl = this.gl;
    this.size = [w, h];
    gl.useProgram(this.prog);
    gl.enableVertexAttribArray(this.loc.pos);
    gl.enableVertexAttribArray(this.loc.uv);
    gl.viewport(0, 0, w, h);
    gl.uniform2f(this.loc.res, w, h);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
  }

  /** Draw state textures of one mesh warped to `dst` (image px), with the given over-alphas. */
  draw(meshKey, dst, draws, texKey = (s) => `${meshKey}/${s}`) {
    const gl = this.gl;
    const m = this.meshes.get(meshKey);
    gl.enableVertexAttribArray(this.loc.pos);  // drawAdaptive may have disabled shared attribute slots
    gl.enableVertexAttribArray(this.loc.uv);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, dst, gl.DYNAMIC_DRAW);
    gl.vertexAttribPointer(this.loc.pos, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, m.idx);
    const alphas = overAlphas(draws);
    draws.forEach(([state], i) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, m.uvs[state]);
      gl.vertexAttribPointer(this.loc.uv, 2, gl.FLOAT, false, 0, 0);
      gl.bindTexture(gl.TEXTURE_2D, this.textures.get(texKey(state)));
      gl.uniform1f(this.loc.alpha, alphas[i]);
      gl.drawElements(gl.TRIANGLES, m.count, gl.UNSIGNED_SHORT, 0);
    });
  }
}

GLMeshRenderer.prototype.drawAdaptive = function drawAdaptive(meshKey, dst, layers, texKey = (s) => `${meshKey}/${s}`) {
  /** One region with the adaptive dissolve (see ADAPTIVE): layers = MorphTrack.sample().layers. */
  const gl = this.gl;
  const m = this.meshes.get(meshKey);
  const L = this.adaptive;
  gl.useProgram(L.p);
  gl.uniform2f(L.u_res, this.size[0], this.size[1]);
  gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
  gl.bufferData(gl.ARRAY_BUFFER, dst, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(L.a_pos);
  gl.vertexAttribPointer(L.a_pos, 2, gl.FLOAT, false, 0, 0);
  const wn = [0, 0, 0], ww = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    const [state, n, w] = layers[Math.min(i, layers.length - 1)];
    if (i < layers.length) { wn[i] = n; ww[i] = w; }
    const a = L[`a_uv${i}`];
    gl.enableVertexAttribArray(a);
    gl.bindBuffer(gl.ARRAY_BUFFER, m.uvs[state]);
    gl.vertexAttribPointer(a, 2, gl.FLOAT, false, 0, 0);
    gl.activeTexture(gl.TEXTURE0 + i);
    gl.bindTexture(gl.TEXTURE_2D, this.textures.get(texKey(state)));
    gl.uniform1i(L[`u_t${i}`], i);
  }
  const [tw, th] = this.texSize.get(texKey(layers[0][0]));
  gl.uniform3f(L.u_wn, ...wn);
  gl.uniform3f(L.u_ww, ...ww);
  gl.uniform2f(L.u_texel, 1 / tw, 1 / th);
  const A = this.adapt ?? ADAPTIVE;  // per avatar (sprite-avatar.js sets it from the manifest's playback settings)
  gl.uniform1f(L.u_r, A.radiusTexels);
  gl.uniform1f(L.u_lo, A.lo);
  gl.uniform1f(L.u_hi, A.hi);
  gl.uniform1f(L.u_slo, A.stretchLo);
  gl.uniform1f(L.u_shi, A.stretchHi);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, m.idx);
  gl.drawElements(gl.TRIANGLES, m.count, gl.UNSIGNED_SHORT, 0);
  for (let i = 0; i < 3; i++) gl.disableVertexAttribArray(L[`a_uv${i}`]);
  gl.activeTexture(gl.TEXTURE0);
  gl.useProgram(this.prog);
};

/** Quad helper (full image, overlay layers): positions + UVs for two triangles. */
export function quad(x, y, w, h) {
  return { tris: [[0, 1, 2], [0, 2, 3]], dst: new Float32Array([x, y, x + w, y, x + w, y + h, x, y + h]),
    uv: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]) };
}

// --- CPU fallback ------------------------------------------------------------------------------------
/**
 * Piecewise-affine warp on a 2D canvas: src = texture px (relative to the texture), dst = image px.
 * Triangles are drawn slightly enlarged so neighbouring triangles leave no hairline gaps.
 */
export function drawMesh2D(ctx, img, src, dst, tris, alpha = 1) {
  ctx.save();
  ctx.globalAlpha = alpha;
  for (const [a, b, c] of tris) {
    const sx0 = src[2 * a], sy0 = src[2 * a + 1], sx1 = src[2 * b], sy1 = src[2 * b + 1], sx2 = src[2 * c], sy2 = src[2 * c + 1];
    let dx0 = dst[2 * a], dy0 = dst[2 * a + 1], dx1 = dst[2 * b], dy1 = dst[2 * b + 1], dx2 = dst[2 * c], dy2 = dst[2 * c + 1];
    const det = (sx1 - sx0) * (sy2 - sy0) - (sx2 - sx0) * (sy1 - sy0);
    if (Math.abs(det) < 1e-6) continue;
    // affine mapping src -> dst
    const ma = ((dx1 - dx0) * (sy2 - sy0) - (dx2 - dx0) * (sy1 - sy0)) / det;
    const mc = ((dx2 - dx0) * (sx1 - sx0) - (dx1 - dx0) * (sx2 - sx0)) / det;
    const mb = ((dy1 - dy0) * (sy2 - sy0) - (dy2 - dy0) * (sy1 - sy0)) / det;
    const md = ((dy2 - dy0) * (sx1 - sx0) - (dy1 - dy0) * (sx2 - sx0)) / det;
    const me = dx0 - ma * sx0 - mc * sy0;
    const mf = dy0 - mb * sx0 - md * sy0;
    const cx = (dx0 + dx1 + dx2) / 3, cy = (dy0 + dy1 + dy2) / 3;
    const grow = (x, y) => { const vx = x - cx, vy = y - cy, l = Math.hypot(vx, vy) || 1; return [x + (vx / l) * 0.6, y + (vy / l) * 0.6]; };
    [dx0, dy0] = grow(dx0, dy0); [dx1, dy1] = grow(dx1, dy1); [dx2, dy2] = grow(dx2, dy2);
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(dx0, dy0); ctx.lineTo(dx1, dy1); ctx.lineTo(dx2, dy2);
    ctx.closePath();
    ctx.clip();
    ctx.setTransform(ma, mb, mc, md, me, mf);
    ctx.drawImage(img, 0, 0);
    ctx.restore();
  }
  ctx.restore();
}
