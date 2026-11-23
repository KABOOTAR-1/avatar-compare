/**
 * Morph core for the 2.5D sprite avatar: landmarks + shared triangles + lerp + draw.
 *
 * Framework-free and DOM-light on purpose, so it ports to React Native (react-native-skia
 * `drawVertices` takes exactly these arrays: positions, texture coordinates, triangle indices):
 *   - MorphTrack: one region's state machine. A transition A -> B moves the mesh from the CURRENT
 *     geometry to B's landmarks with an ease-in-out and dissolves the textures; retargeting mid-way
 *     starts from where the mesh is (no geometric pop; acts as a low-pass on the jaw).
 *   - GLMeshRenderer: WebGL1, one draw call per texture per region, precomputed UV buffers.
 *   - drawMesh2D: CPU canvas fallback (per-triangle affine + clip).
 * Data comes from scripts/avatar3d/sprite/morph_data.py (morph.json): per region a box, a triangle
 * list shared by all states, and per state a texture (same alpha for all states) + points.
 */

export const easeInOut = (t) => (t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t));
const clamp01 = (x) => Math.min(1, Math.max(0, x));
const MIN_WEIGHT = 0.002;
const MAX_DRAWS = 3;

export function lerpPoints(a, b, t, out = new Float32Array(a.length)) {
  for (let i = 0; i < a.length; i++) out[i] = a[i] + (b[i] - a[i]) * t;
  return out;
}

/** One region: current geometry + texture weights, retargetable at any time. */
export class MorphTrack {
  /** @param {Object<string, Float32Array>} points state id -> flat [x0,y0,x1,y1,...] (image px) */
  constructor(points, initial = "base") {
    this.points = points;
    this.to = initial;
    this.from = { points: points[initial].slice(), mix: [[initial, 1]] };
    this.start = 0;
    this.dur = 1;
    this.cur = points[initial].slice();
  }

  /** Move towards `target` over `durMs` (eased), starting from the geometry shown at `now`. */
  set(target, now, durMs) {
    if (!(target in this.points) || target === this.to) return;
    const snap = this.sample(now);
    this.from = { points: snap.points.slice(), mix: snap.draws.map(([k, w]) => [k, w]) };
    this.to = target;
    this.start = now;
    this.dur = Math.max(1, durMs);
  }

  /** {points (Float32Array, image px), draws: [[stateId, weight], ...] summing to 1, largest first} */
  sample(now) {
    const u = easeInOut(clamp01((now - this.start) / this.dur));
    lerpPoints(this.from.points, this.points[this.to], u, this.cur);
    const w = new Map();
    for (const [k, v] of this.from.mix) w.set(k, (w.get(k) ?? 0) + v * (1 - u));
    w.set(this.to, (w.get(this.to) ?? 0) + u);
    let draws = [...w.entries()].filter(([, v]) => v > MIN_WEIGHT).sort((a, b) => b[1] - a[1]).slice(0, MAX_DRAWS);
    const sum = draws.reduce((s, [, v]) => s + v, 0) || 1;
    draws = draws.map(([k, v]) => [k, v / sum]);
    return { points: this.cur, draws };
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
    this.meshes = new Map();
    this.posBuf = gl.createBuffer();
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
    gl.viewport(0, 0, w, h);
    gl.uniform2f(this.loc.res, w, h);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
  }

  /** Draw state textures of one mesh warped to `dst` (image px), with the given over-alphas. */
  draw(meshKey, dst, draws, texKey = (s) => `${meshKey}/${s}`) {
    const gl = this.gl;
    const m = this.meshes.get(meshKey);
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
