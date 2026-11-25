/**
 * SpriteAvatar: 2.5D talking head from layered Gemini sprites (see
 * scripts/avatar3d/sprite/build_layers.py for how the layers are cut).
 *
 * Per frame: compose face.png + brows + eyes + mouth + glasses into an offscreen canvas (each
 * region cross-fades independently, so blinks run while talking), then split it into a static
 * body plate and a head cut-out that gets a small 2D affine sway/nod around the neck, and apply
 * a breathing scale to both. Input signals are the same as a 3D avatar's: an Oculus viseme
 * timeline, an expression name, gaze, blink.
 *
 * Azure 22-viseme set: when the manifest has "visemes_azure" (id -> mouth layer, or {glide: [a, b]}
 * for the diphthongs 9/10/11) AND the clip has a "visemes_azure" timeline, that richer timeline
 * drives the mouth; a diphthong holds its first shape, then cross-fades into the second over the
 * rest of its cue. Otherwise the 15 Oculus visemes are used exactly as before.
 *
 * Morph mode (manifest "morph": morph.json from morph_data.py): no alpha cross-fades at all. Each
 * region shows ONE full texture over the same mask at rest; a change A -> B warps the mesh from the
 * current geometry to B's landmarks (ease-in-out, shared Delaunay triangles) and dissolves the two
 * aligned textures, rendered with WebGL (morph-core.js; CPU canvas fallback). Without morph.json the
 * cross-fade path below runs unchanged.
 */
import { MorphTrack, GLMeshRenderer, drawMesh2D, overAlphas, quad, TIMELINE_SIGMA_S, TIMELINE_CONTACT_SIGMA_S,
  SWITCH_WINDOW, ADAPTIVE, TIMELINE_SHARPEN } from "./morph-core.js";

// --- timing (ms) -----------------------------------------------------------------------------
export const VISEME_FADE_MS = 70;
const EXPRESSION_FADE_MS = 180;
const GAZE_FADE_MS = 90;
const BLINK_STEP_FADE_MS = 25;
const BLINK_SEQUENCE = [["half", 40], ["closed", 70], ["half", 45]]; // eye state, hold ms
const BLINK_INTERVAL_MS = [2200, 5600];
const DOUBLE_BLINK_CHANCE = 0.18;
const GLANCE_INTERVAL_MS = [3500, 8000];
const GLANCE_HOLD_MS = [500, 1300];
const VISEME_LEAD_S = 0.03; // mouth shapes lead the audio slightly, as in hand animation
const GLIDE_HOLD = 0.3;     // diphthong: share of the cue that holds the first shape before gliding
const GLIDE_PREVIEW_MS = 450;
// morph timing (ms)
const MORPH_VISEME_MS = [40, 120];   // transition = clamp(share x phone duration, min, max)
const MORPH_VISEME_SHARE = 0.6;
// speech plays viseme shapes at this share of their full pose (see MorphTrack.set). 0.7 was for the exaggerated
// image-model poses; the shapes are now the photo moved by LivePortrait (natural size), and 0.7 left the mouth
// half-open while talking.
const SPEECH_MOUTH_AMOUNT = 1.0;
// Playback settings: these are only the fallbacks. Each avatar's manifest "playback" (chosen per avatar by the AI judge
// watching captioned talking clips, scripts/avatar3d/sprite/tune_playback.py) overrides them.
const PLAYBACK_DEFAULTS = { timeline: true, sigma_s: TIMELINE_SIGMA_S, contact_sigma_s: TIMELINE_CONTACT_SIGMA_S,
  speech_amount: SPEECH_MOUTH_AMOUNT, lead_s: VISEME_LEAD_S, switch_window: SWITCH_WINDOW, sharpen: TIMELINE_SHARPEN,
  adapt_lo: ADAPTIVE.lo, adapt_hi: ADAPTIVE.hi, adapt_radius: ADAPTIVE.radiusTexels, stretch_lo: ADAPTIVE.stretchLo,
  stretch_hi: ADAPTIVE.stretchHi, undershoot_s: 0 };
const MIN_CUE_S = 0.017;             // shapes shorter than about one frame are skipped (coarticulation)
const MORPH_EXPRESSION_MS = 280;
const MORPH_GAZE_MS = 240;
const MORPH_BLINK_CLOSE_MS = 70;
const MORPH_BLINK_OPEN_MS = 150;
const MORPH_BLINK_SEQUENCE = [["closed", 110]]; // close (70 ms morph) + short hold, then open (150 ms)
const MICRO_PX = 0.35;               // head micro-motion amplitude (sub-pixel, whole image)

// --- procedural motion -----------------------------------------------------------------------
const BREATH_PERIOD_S = 4.2;
const BREATH_SCALE = 0.006;      // y-scale amplitude about the bottom edge
const SWAY_DEG = 0.9;            // head roll amplitude
const SWAY_PX = 1.5;             // head drift amplitude (image px)
const TALK_NOD_DEG = 0.6;        // extra nodding while talking
const NECK_FEATHER_PX = 14;      // head/body blend band around the neck cut (kept below the jaw)
const CHIN_CLEAR_PX = 3;          // extra gap between the lowest chin and the blend band
// Head sway as ONE smooth bend of the picture (not a head layer blended into the body at a band: anything crossing
// that band, the neck outline or a collar, showed twice while the head moved, ai-curly / ai-gym). Above the lowest
// chin the head moves fully; down the neck the motion fades out; below that nothing moves. Nothing is drawn twice.
const BEND_NECK_X = 2.0;          // fade length = this x (neck pivot - chin): roughly chin to shoulders
const BEND_STRIP_PX = 3;          // strip height inside the fade (each strip its own share of the motion)
const BEND_OVERLAP_PX = 4;        // strips overlap the already-drawn strip below them by this much
const BEND_MIN_PX = 4;            // shortest fade (a collar right under the chin)
const NECK_CUT_BELOW_CHIN = 0.53; // cut line: chin + this x (pivot - chin); below the dropped jaw of "aa", above the collar
const BG_MATTE_LO = 6;           // colour distance from the flat background that counts as figure
const HEAD_MATTE_MARGIN_PX = 6;  // matte grown into pure background so its edge is invisible

const REGIONS = ["brows", "eyes", "mouth"];

const loadImage = (src) => new Promise((resolve, reject) => {
  const img = new Image();
  img.crossOrigin = "anonymous"; // layers may come from a CDN; getImageData needs an untainted canvas
  img.onload = () => resolve(img);
  img.onerror = () => reject(new Error(`failed to load ${src}`));
  img.src = src;
});

const rand = ([a, b]) => a + Math.random() * (b - a);
const clamp01 = (x) => Math.min(1, Math.max(0, x));
/** Binary dilation with a square (2r+1) kernel, separable. */
function dilate(src, w, h, r) {
  const tmp = new Uint8Array(w * h);
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!src[y * w + x]) continue;
      for (let k = Math.max(0, x - r); k <= Math.min(w - 1, x + r); k++) tmp[y * w + k] = 1;
    }
  }
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      if (!tmp[y * w + x]) continue;
      for (let k = Math.max(0, y - r); k <= Math.min(h - 1, y + r); k++) out[k * w + x] = 1;
    }
  }
  return out;
}
const smoothstep = (lo, hi, x) => { const t = clamp01((x - lo) / (hi - lo)); return t * t * (3 - 2 * t); };

/** One region's cross-fade state: the incoming layer id and the outgoing one. */
class RegionFade {
  constructor() { this.cur = null; this.prev = null; this.start = 0; this.dur = 1; }
  set(id, now, dur) {
    if (id === this.cur) return;
    this.prev = this.cur;
    this.cur = id;
    this.start = now;
    this.dur = Math.max(1, dur);
  }
  /** [[id, alpha], ...] to draw in order; both stay opaque through the middle of the fade. */
  draws(now) {
    const k = clamp01((now - this.start) / this.dur);
    const out = [];
    if (this.prev && k < 1) out.push([this.prev, Math.min(1, 2 * (1 - k))]);
    if (this.cur) out.push([this.cur, Math.min(1, 2 * k)]);
    return out;
  }
}

// MediaPipe landmark ids (the face model's fixed topology, the same for every face): inner-lip centre and corners
const LIP_IN_TOP = 13, LIP_IN_BOTTOM = 14, LIP_IN_L = 78, LIP_IN_R = 308;
const CONTACT_GAP = 0.03;  // inner-lip gap / inner mouth width below this = the lips touch (= morph_render.py)

/** States of a morph region whose lips touch, measured on each state's own points (no state names). */
function contactStates(r, points) {
  const idx = (id) => { const j = (r.landmarks ?? []).indexOf(id); return j < 0 ? -1 : r.ring + j; };
  const [a, b, l, rr] = [LIP_IN_TOP, LIP_IN_BOTTOM, LIP_IN_L, LIP_IN_R].map(idx);
  if ([a, b, l, rr].some((i) => i < 0)) return new Set();
  const d = (p, i, j) => Math.hypot(p[2 * i] - p[2 * j], p[2 * i + 1] - p[2 * j + 1]);
  return new Set(Object.entries(points).filter(([, p]) => d(p, a, b) < CONTACT_GAP * d(p, l, rr)).map(([s]) => s));
}

export class SpriteAvatar {
  /** @param {HTMLCanvasElement} canvas  @param {string} root URL of the avatar dir (ends with /) */
  constructor(canvas, root) {
    this.canvas = canvas;
    this.root = root;
    this.ctx = canvas.getContext("2d");
    this.fades = Object.fromEntries(REGIONS.map((r) => [r, new RegionFade()]));
    this.expression = "neutral";
    this.gaze = null;          // manual gaze ("left" | "right" | "up" | null)
    this.idleGaze = null;      // automatic glance
    this.blinkState = null;    // "half" | "closed" | null
    this.mouthOverride = null; // viseme preview from the UI
    this.speech = null;        // { clip, audio }
    this.life = true;
    this.autoBlink = true;
    this.autoGlance = true;
    this.showBounds = false;
    this.visemeFadeMs = VISEME_FADE_MS;
    this.current = { viseme: "sil", phone: "", word: "" };
    this.listeners = new Set();
  }

  async load() {
    // a fresh tag on every file: a picture cached while a rebuild was half-written (a black face) must never be
    // reused; a hard refresh did not re-fetch pictures the script loads (ai-curly-lp)
    this.bust ??= `?v=${Date.now()}`;
    this.manifest = await (await fetch(`${this.root}manifest.json${this.bust}`, { cache: "no-store" })).json();
    const m = this.manifest;
    const files = { face: m.base };
    if (m.matte) files.matte = m.matte;  // soft person matte (head_matte.py, MODNet); else the colour test below
    for (const [region, layers] of Object.entries(m.layers)) {
      for (const [id, layer] of Object.entries(layers)) files[`${region}/${id}`] = layer.file;
    }
    const entries = await Promise.all(Object.entries(files).map(async ([k, f]) => [k, await loadImage(this.root + f + this.bust)]));
    this.images = Object.fromEntries(entries);
    if (m.morph) {
      try {
        await this.loadMorph(m.morph);
      } catch (error) {
        console.warn("morph data unavailable, using cross-fades", error);
        this.morph = null;
      }
    }
    const { width: w, height: h } = m;
    this.canvas.width = w;
    this.canvas.height = h;
    const mk = () => Object.assign(document.createElement("canvas"), { width: w, height: h });
    this.faceCanvas = mk();
    this.plateCanvas = mk();
    this.headCanvas = mk();
    this.buildHeadMasks();
    this.scheduleBlink(performance.now());
    this.scheduleGlance(performance.now());
    this.loop = this.loop.bind(this);
    requestAnimationFrame(this.loop);
  }

  /**
   * Head matte for the 2D sway: everything that differs from the flat background in the face OR
   * in any layer (a dropped jaw is wider/longer than the base's), grown by a margin of pure
   * background so the matte edge never crosses the silhouette; cut softly at the neck.
   */
  buildHeadMasks() {
    const { width: w, height: h, anchors, layers } = this.manifest;
    const c = Object.assign(document.createElement("canvas"), { width: w, height: h });
    const g = c.getContext("2d", { willReadFrequently: true });
    g.drawImage(this.images.face, 0, 0);
    const base = g.getImageData(0, 0, w, h).data;
    const bg = [base[0], base[1], base[2]];
    this.bgColor = `rgb(${bg.join(",")})`;
    // person alpha 0..255: the build-time matte when there is one (pixel-accurate, soft hair edges), else the old
    // colour test (distance from the top-left pixel) grown by a margin
    let person;
    if (this.images.matte) {
      g.drawImage(this.images.matte, 0, 0, w, h);
      const md = g.getImageData(0, 0, w, h).data;
      person = new Uint8Array(w * h);
      for (let k = 0; k < w * h; k++) person[k] = md[k * 4];
      g.drawImage(this.images.face, 0, 0);
    }
    const fg = new Uint8Array(w * h);
    const mark = (d, x0, y0, cw, ch) => {
      for (let y = 0; y < ch; y++) {
        for (let x = 0; x < cw; x++) {
          const i = (y * cw + x) * 4;
          const dist = Math.max(Math.abs(d[i] - bg[0]), Math.abs(d[i + 1] - bg[1]), Math.abs(d[i + 2] - bg[2]));
          if (dist > BG_MATTE_LO) fg[(y + y0) * w + x + x0] = 1;
        }
      }
    };
    mark(base, 0, 0, w, h);
    for (const [region, items] of Object.entries(layers)) {
      for (const [id, l] of Object.entries(items)) {
        g.drawImage(this.images.face, l.x, l.y, l.w, l.h, l.x, l.y, l.w, l.h);
        g.drawImage(this.images[`${region}/${id}`], l.x, l.y);
        mark(g.getImageData(l.x, l.y, l.w, l.h).data, l.x, l.y, l.w, l.h);
      }
    }
    if (!person) {
      const grown = dilate(fg, w, h, HEAD_MATTE_MARGIN_PX);
      person = new Uint8Array(w * h);
      for (let k = 0; k < w * h; k++) person[k] = grown[k] ? 255 : 0;
    }
    const chinY = anchors.chin[1];
    // below the LOWEST chin of any frame (+ the blend band): else an open jaw's chin bottom stays with the still
    // body while the head sways, and the collar peeks through (build_layers.py chin_low)
    const cutY = Math.max(chinY + NECK_CUT_BELOW_CHIN * (anchors.neck_pivot[1] - chinY),
      (anchors.chin_low?.[1] ?? 0) + NECK_FEATHER_PX + CHIN_CLEAR_PX);
    this.pivot = [anchors.neck_pivot[0], cutY];
    const head = g.createImageData(w, h);
    const clear = g.createImageData(w, h);
    for (let y = 0; y < h; y++) {
      const v = 1 - smoothstep(cutY - NECK_FEATHER_PX, cutY + NECK_FEATHER_PX, y);
      const above = y < cutY - NECK_FEATHER_PX;
      for (let x = 0; x < w; x++) {
        const k = y * w + x;
        head.data[k * 4 + 3] = Math.round(person[k] * v);
        clear.data[k * 4 + 3] = above ? person[k] : 0;
      }
    }
    const toCanvas = (img) => {
      const k = Object.assign(document.createElement("canvas"), { width: w, height: h });
      k.getContext("2d").putImageData(img, 0, 0);
      return k;
    };
    this.headMask = toCanvas(head);
    this.clearMask = toCanvas(clear);
    const chinLow = anchors.chin_low?.[1] ?? chinY;
    this.bendTop = Math.round(chinLow + CHIN_CLEAR_PX);
    this.bendBottom = Math.min(h, Math.round(this.bendTop + BEND_NECK_X * Math.max(20, anchors.neck_pivot[1] - chinY)));
    // clothing never sways with the head: the bend ends where the clothing starts (a turtleneck right under the chin:
    // the head moves above the chin and the sweater stays still)
    if (anchors.clothes_top) this.bendBottom = Math.max(this.bendTop + BEND_MIN_PX, Math.min(this.bendBottom, Math.round(anchors.clothes_top)));
    this.bendPivot = [anchors.neck_pivot[0], this.bendBottom];
  }

  /** morph.json -> textures, per-state points / UVs, one MorphTrack per region, WebGL (or 2D) renderer. */
  async loadMorph(file) {
    const data = await (await fetch(this.root + file + this.bust)).json();
    const { width: w, height: h } = this.manifest;
    const regions = {};
    for (const [name, r] of Object.entries(data.regions)) {
      const ids = Object.keys(r.states);
      const imgs = await Promise.all(ids.map((s) => loadImage(this.root + r.states[s].file + this.bust)));
      const points = {}, src = {}, uv = {}, img = {};
      ids.forEach((s, i) => {
        const p = Float32Array.from(r.states[s].points.flat());
        points[s] = p;
        src[s] = p.map((v, k) => v - (k % 2 ? r.y : r.x));
        uv[s] = p.map((v, k) => (k % 2 ? (v - r.y) / r.h : (v - r.x) / r.w));
        img[s] = imgs[i];
      });
      regions[name] = { ...r, points, src, uv, img, track: new MorphTrack(points, r.base ?? "base"),
        contact: contactStates(r, points) };
    }
    const canvas = Object.assign(document.createElement("canvas"), { width: w, height: h });
    let gl = null;
    try {
      gl = new GLMeshRenderer(canvas);
      gl.texture("face", this.images.face);
      const fq = quad(0, 0, w, h);
      gl.mesh("face", fq.tris, { face: fq.uv });
      for (const [name, r] of Object.entries(regions)) {
        for (const [s, im] of Object.entries(r.img)) gl.texture(`${name}/${s}`, im);
        gl.mesh(name, r.triangles, r.uv);
      }
      for (const [region, id] of Object.entries(this.manifest.overlay ?? {})) {
        const l = this.manifest.layers[region][id];
        const q = quad(l.x, l.y, l.w, l.h);
        gl.texture(`overlay/${region}`, this.images[`${region}/${id}`]);
        gl.mesh(`overlay/${region}`, q.tris, { [region]: q.uv });
        regions[`overlay/${region}`] = { overlay: true, dst: q.dst, state: region };
      }
    } catch (error) {
      console.warn("WebGL unavailable, CPU canvas morph", error);
      gl = null;
    }
    const play = this.playback();
    for (const r of Object.values(regions)) if (r.track) r.track.switchWindow = play.switch_window;
    if (gl) gl.adapt = { radiusTexels: play.adapt_radius, lo: play.adapt_lo, hi: play.adapt_hi, stretchLo: play.stretch_lo,
      stretchHi: play.stretch_hi };
    this.morph = { regions, canvas, gl, ctx2d: gl ? null : canvas.getContext("2d") };
    this.renderer = gl ? "webgl" : "canvas2d";
  }

  /**
   * Morph targets: brows and eyes form ONE region ("upper", state "<brows>|<eyes>", textures made from
   * exactly that pair), the mouth another. {brows: [id, ms], eyes: [id, ms], mouth: [id, ms]}.
   */
  setMorphTargets(tg, now) {
    const r = this.morph.regions;
    if (r.upper) {
      const [b, bms] = tg.brows ?? [null, MORPH_EXPRESSION_MS];
      const [e, ems] = tg.eyes ?? [null, MORPH_EXPRESSION_MS];
      const key = `${b ?? "base"}|${e ?? "base"}`;
      const [cb, ce] = r.upper.track.to.split("|");
      const ms = (ce !== (e ?? "base")) ? ems : bms;  // an eye change (blink, gaze) sets the pace
      r.upper.track.set(key, now, cb !== (b ?? "base") && ce !== (e ?? "base") ? Math.max(bms, ems) : ms);
    }
    if (r.mouth && tg.mouth) r.mouth.track.set(tg.mouth[0] ?? "base", now, tg.mouth[1], tg.mouth[2] ?? 1);
  }

  /** Render the face with every region at its current morph state into this.faceCanvas. */
  composeMorph(now) {
    const { regions, canvas, gl, ctx2d } = this.morph;
    const { width: w, height: h } = this.manifest;
    this.bounds = [];
    if (gl) {
      gl.begin(w, h);
      gl.draw("face", quad(0, 0, w, h).dst, [["face", 1]], () => "face");
    } else {
      ctx2d.globalAlpha = 1;
      ctx2d.drawImage(this.images.face, 0, 0);
    }
    for (const name of ["mouth", "upper", "brows", "eyes"]) {  // upper drawn over the mouth region's top edge
      const r = regions[name];
      if (!r || r.overlay) continue;
      const { points, draws, layers } = r.track.sample(now);
      if (gl) gl.drawAdaptive(name, points, layers);
      else {
        const alphas = overAlphas(draws);
        draws.forEach(([s], i) => drawMesh2D(ctx2d, r.img[s], r.src[s], points, r.triangles, alphas[i]));
      }
      this.bounds.push([name, { x: r.x, y: r.y, w: r.w, h: r.h }]);
    }
    for (const [key, r] of Object.entries(regions)) {
      if (!r.overlay) continue;
      if (gl) gl.draw(key, r.dst, [[r.state, 1]], () => key);
      else {
        const [region, id] = [r.state, this.manifest.overlay[r.state]];
        const l = this.manifest.layers[region][id];
        ctx2d.drawImage(this.images[`${region}/${id}`], l.x, l.y);
      }
    }
    const g = this.faceCanvas.getContext("2d");
    g.globalAlpha = 1;
    g.drawImage(canvas, 0, 0);
  }

  // --- public controls ------------------------------------------------------------------------
  setExpression(name) { this.expression = name in this.manifest.expressions ? name : "neutral"; }
  setGaze(dir) { this.gaze = dir; }
  previewViseme(v) { this.mouthOverride = v; this.overrideStart = performance.now(); }
  blink(now = performance.now()) { this.blinkQueue = this.makeBlink(now, false); }
  onUpdate(fn) { this.listeners.add(fn); }

  /** Play a speech clip: {audio url, visemes:[{start,end,viseme,phone}], words, expression}. */
  async say(clip) {
    this.stopSpeech();
    const audio = new Audio(this.root + clip.audio);
    this.speech = { clip, audio };
    const mouth = this.morph?.regions.mouth;
    const play = this.playback();
    if (mouth && play.timeline) {  // the whole shape sequence is known: the mouth follows it with zero-lag smoothing
      const speech = this.speech;
      mouth.track.setTimeline(this.speechKeys(clip), () => speech.audio.currentTime + play.lead_s, mouth.contact,
        { sigma: play.sigma_s, contactSigma: play.contact_sigma_s, amount: play.speech_amount, rest: this.speechRestMouth(),
          sharpen: play.sharpen, undershoot: play.undershoot_s });
    }
    audio.addEventListener("ended", () => { if (this.speech?.audio === audio) this.stopSpeech(); });
    try {
      await audio.play();
    } catch (error) {
      console.error("audio play failed", error);
      this.stopSpeech();
    }
  }

  stopSpeech() {
    if (this.speech) this.speech.audio.pause();
    this.speech = null;
    const mouth = this.morph?.regions.mouth;
    if (mouth?.track.timeline) mouth.track.clearTimeline(performance.now(), this.restMouth(), MORPH_EXPRESSION_MS);
  }

  /** This avatar's playback settings (manifest "playback" over PLAYBACK_DEFAULTS). */
  playback() {
    return { ...PLAYBACK_DEFAULTS, ...(this.manifest?.playback?.settings ?? {}) };
  }

  /**
   * The mouth between words while talking: always the neutral mouth. The line's expression stays in the eyes and brows
   * (a happy line has smiling eyes); with the expression's mouth here (a smile) the mouth snapped back to the smile in
   * every pause between words.
   */
  speechRestMouth() {
    return this.morph?.regions.mouth?.base ?? "base";
  }

  /** The mouth shown when not talking: the expression's mouth, else the base. */
  restMouth() {
    const expr = this.manifest.expressions[this.expression] ?? this.manifest.expressions.neutral;
    return expr.mouth ?? "base";
  }

  /**
   * A clip's mouth shapes as contiguous keys [{start, end, state}] (seconds): Azure visemes when the avatar has them
   * (a diphthong = its two shapes, split at GLIDE_HOLD), else the Oculus ones; silence and unmapped ids = the rest
   * mouth, which also fills gaps and runs before and after the clip.
   */
  speechKeys(clip) {
    const m = this.manifest, rest = this.speechRestMouth(), keys = [];
    const push = (start, end, state) => {
      if (end <= start) return;
      const last = keys[keys.length - 1];
      if (last && last.state === state && Math.abs(last.end - start) < 1e-6) last.end = end;
      else keys.push({ start, end, state });
    };
    const azure = m.visemes_azure && clip.visemes_azure;
    const cues = azure ? clip.visemes_azure : clip.visemes;
    let t = -1e3;
    for (const c of cues) {
      push(t, c.start, rest);
      const entry = azure ? m.visemes_azure[String(c.id)] : m.visemes[c.viseme];
      if (entry && typeof entry === "object" && entry.glide) {
        const split = c.start + GLIDE_HOLD * (c.end - c.start);
        push(c.start, split, entry.glide[0] ?? rest);
        push(split, c.end, entry.glide[1] ?? rest);
      } else push(c.start, c.end, entry ?? rest);
      t = c.end;
    }
    push(t, t + 1e3, rest);
    return keys;
  }

  // --- state machine --------------------------------------------------------------------------
  makeBlink(now, allowDouble = true) {
    const steps = [];
    let t = now;
    const reps = allowDouble && Math.random() < DOUBLE_BLINK_CHANCE ? 2 : 1;
    for (let r = 0; r < reps; r++) {
      for (const [state, ms] of this.morph ? MORPH_BLINK_SEQUENCE : BLINK_SEQUENCE) { steps.push({ state, at: t }); t += ms; }
      steps.push({ state: null, at: t });
      t += this.morph ? MORPH_BLINK_OPEN_MS + 60 : 90;
    }
    return steps;
  }

  scheduleBlink(now) { this.nextBlinkAt = now + rand(BLINK_INTERVAL_MS); }
  scheduleGlance(now) { this.nextGlanceAt = now + rand(GLANCE_INTERVAL_MS); }

  updateEyesAuto(now) {
    if (this.autoBlink && !this.blinkQueue && now >= this.nextBlinkAt) {
      this.blinkQueue = this.makeBlink(now);
      this.scheduleBlink(now);
    }
    if (this.blinkQueue) {
      while (this.blinkQueue.length && this.blinkQueue[0].at <= now) this.blinkState = this.blinkQueue.shift().state;
      if (!this.blinkQueue.length) this.blinkQueue = null;
    }
    if (this.autoGlance && !this.speech && now >= this.nextGlanceAt) {
      if (this.idleGaze) { this.idleGaze = null; this.scheduleGlance(now); }
      else {
        const dirs = Object.keys(this.manifest.eyes.gaze);
        this.idleGaze = dirs[Math.floor(Math.random() * dirs.length)];
        this.nextGlanceAt = now + rand(GLANCE_HOLD_MS);
      }
    }
  }

  currentViseme() {
    if (!this.speech) return null;
    const t = this.speech.audio.currentTime + this.playback().lead_s;
    const { visemes, words } = this.speech.clip;
    const word = words?.find((w) => t >= w.start && t < w.end);
    const azure = this.manifest.visemes_azure && this.speech.clip.visemes_azure;
    if (azure && this.morph) {
      // coarticulation: drop sub-frame cues, and start moving to the next shape half a transition early
      const cues = (this.speech.clip._cues ??= azure.filter((c) => c.end - c.start >= MIN_CUE_S));
      let i = cues.findIndex((c) => t >= c.start && t < c.end);
      if (i < 0) i = t < cues[0].start ? 0 : cues.length - 1;
      const next = cues[i + 1];
      if (next) {
        const nextMs = Math.min(MORPH_VISEME_MS[1], Math.max(MORPH_VISEME_MS[0], MORPH_VISEME_SHARE * (next.end - next.start) * 1000));
        if (t + nextMs / 2000 >= next.start) i += 1;
      }
      const cue = cues[i];
      const dur = Math.max(cue.end - cue.start, 1e-3);
      return { viseme: `az${cue.id}`, azure: String(cue.id), phone: cue.phone, word: word?.word ?? "",
        progress: clamp01((t - cue.start) / dur), durMs: dur * 1000,
        morphMs: Math.min(MORPH_VISEME_MS[1], Math.max(MORPH_VISEME_MS[0], MORPH_VISEME_SHARE * dur * 1000)) };
    }
    if (azure) {
      const cue = azure.find((c) => t >= c.start && t < c.end) ?? azure[azure.length - 1];
      const dur = Math.max(cue.end - cue.start, 1e-3);
      return { viseme: `az${cue.id}`, azure: String(cue.id), phone: cue.phone, word: word?.word ?? "",
        progress: clamp01((t - cue.start) / dur), durMs: dur * 1000 };
    }
    const cue = visemes.find((c) => t >= c.start && t < c.end) ?? visemes[visemes.length - 1];
    const ms = Math.min(MORPH_VISEME_MS[1], Math.max(MORPH_VISEME_MS[0], MORPH_VISEME_SHARE * (cue.end - cue.start) * 1000));
    return { viseme: cue.viseme, phone: cue.phone, word: word?.word ?? "", morphMs: ms };
  }

  /** Mouth layer (+ fade ms or null for the default) for an Azure viseme id at progress 0..1. */
  azureMouth(id, progress, durMs) {
    const entry = this.manifest.visemes_azure[id];
    if (!entry || typeof entry === "string") return [entry ?? null, null];
    const [from, to] = entry.glide;
    if (progress < GLIDE_HOLD) return [from, null];
    return [to, Math.max(1, (1 - GLIDE_HOLD) * durMs)];  // glide: cross-fade over the rest of the cue
  }

  /** Decide each region's target layer id. */
  targets(now) {
    const m = this.manifest;
    const expr = m.expressions[this.expression] ?? m.expressions.neutral;
    const speaking = this.currentViseme();
    let mouth = expr.mouth;
    let viseme = "sil";
    let glideFade = null;
    if (speaking) {
      viseme = speaking.viseme;
      if (speaking.azure !== undefined) [mouth, glideFade] = this.azureMouth(speaking.azure, speaking.progress, speaking.durMs);
      else mouth = m.visemes[viseme] ?? null;
    }
    if (this.mouthOverride) {
      viseme = this.mouthOverride;
      if (viseme.startsWith("az:") && m.visemes_azure) {
        const p = clamp01((now - (this.overrideStart ?? now)) / GLIDE_PREVIEW_MS);
        [mouth, glideFade] = this.azureMouth(viseme.slice(3), p, GLIDE_PREVIEW_MS);
      } else mouth = m.visemes[viseme] ?? null;
    }
    this.current = { viseme, phone: speaking?.phone ?? "", word: speaking?.word ?? "" };

    // eyes: blink > gaze > expression. Expressions whose own eyes are shut (measured at build time,
    // manifest "eyes_closed") do not blink.
    const eyesClosedByExpr = Boolean(expr.eyes_closed);
    let eyes = expr.eyes;
    const gaze = this.gaze ?? (this.expression === "neutral" ? this.idleGaze : null);
    if (gaze) eyes = m.eyes.gaze[gaze];
    let eyeFade = gaze ? GAZE_FADE_MS : EXPRESSION_FADE_MS;
    if (this.blinkState && !eyesClosedByExpr) {
      eyes = this.blinkState === "closed" ? m.eyes.closed : m.eyes.half;
      eyeFade = BLINK_STEP_FADE_MS;
    } else if (this.fades.eyes.cur === m.eyes.half || this.fades.eyes.cur === m.eyes.closed) {
      eyeFade = BLINK_STEP_FADE_MS;
    }
    const mouthFade = glideFade ?? (speaking || this.mouthOverride ? this.visemeFadeMs : EXPRESSION_FADE_MS);
    if (this.morph) {
      const closing = this.blinkState === "closed";
      const fromBlink = this.morph.regions.upper?.track.to.endsWith(`|${m.eyes.closed}`) && !closing;
      const eyeMs = closing ? MORPH_BLINK_CLOSE_MS : fromBlink ? MORPH_BLINK_OPEN_MS : gaze ? MORPH_GAZE_MS : MORPH_EXPRESSION_MS;
      const mouthMs = glideFade ?? (speaking ? speaking.morphMs ?? MORPH_VISEME_MS[1] : this.mouthOverride ? MORPH_VISEME_MS[1] : MORPH_EXPRESSION_MS);
      const mouthAmount = speaking && !this.mouthOverride && mouth ? this.playback().speech_amount : 1;
      return { brows: [expr.brows, MORPH_EXPRESSION_MS], eyes: [eyes, eyeMs], mouth: [mouth, mouthMs, mouthAmount] };
    }
    return {
      brows: [expr.brows, EXPRESSION_FADE_MS],
      eyes: [eyes, eyeFade],
      mouth: [mouth, mouthFade],
    };
  }

  // --- rendering ------------------------------------------------------------------------------
  composeFace(now) {
    const g = this.faceCanvas.getContext("2d");
    const m = this.manifest;
    g.globalAlpha = 1;
    g.drawImage(this.images.face, 0, 0);
    this.bounds = [];
    for (const region of m.draw_order) {
      if (region === "glasses") continue;
      for (const [id, alpha] of this.fades[region].draws(now)) {
        const layer = m.layers[region][id];
        if (!layer || alpha <= 0) continue;
        g.globalAlpha = alpha;
        g.drawImage(this.images[`${region}/${id}`], layer.x, layer.y);
        this.bounds.push([region, layer]);
      }
    }
    g.globalAlpha = 1;
    for (const [region, id] of Object.entries(m.overlay ?? {})) {
      const layer = m.layers[region][id];
      g.drawImage(this.images[`${region}/${id}`], layer.x, layer.y);
    }
  }

  draw(now) {
    const { width: w, height: h } = this.manifest;
    if (this.morph) this.composeMorph(now);
    else this.composeFace(now);
    const t = now / 1000;
    const talking = this.speech ? 1 : 0;
    // smooth pseudo-noise from incommensurate sines
    const n1 = Math.sin(t * 0.73) * 0.6 + Math.sin(t * 1.31 + 1.7) * 0.4;
    const n2 = Math.sin(t * 0.57 + 0.4) * 0.6 + Math.sin(t * 1.13 + 2.9) * 0.4;
    const n3 = Math.sin(t * 2.3) * 0.5 + Math.sin(t * 3.7 + 0.8) * 0.5;
    const life = this.life ? 1 : 0;
    const breath = life * Math.sin((2 * Math.PI * t) / BREATH_PERIOD_S);
    const roll = life * (n1 * SWAY_DEG + talking * n3 * TALK_NOD_DEG * 0.5) * (Math.PI / 180);
    // + subtle head micro-motion: slow noise, sub-pixel, whole image
    const micro = this.morph ? life * MICRO_PX : 0;
    const dx = life * n2 * SWAY_PX + micro * Math.sin(t * 0.37 + 1.1) * Math.sin(t * 0.23);
    const dy = life * (n1 * SWAY_PX * 0.4 + talking * Math.abs(n3) * 1.5) + micro * Math.sin(t * 0.29 + 2.3);

    const c = this.ctx;
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.fillStyle = this.bgColor;
    c.fillRect(0, 0, w, h);
    // breathing: scale about the bottom edge (shoulders rise, head rides along)
    const s = 1 + BREATH_SCALE * breath;
    const breathe = () => c.setTransform(1, 0, 0, s, 0, h * (1 - s));
    // the bend: a share k of the head's motion (1 = the head, 0 = the still body), one strip of rows at a time
    const [bx, by] = this.bendPivot;
    const strip = (y0, y1, k) => {
      breathe();
      if (k > 0) {
        c.translate(bx + dx * k, by + dy * k);
        c.rotate(roll * k);
        c.translate(-bx, -by);
      }
      // each strip reaches BEND_OVERLAP_PX down over the one drawn before it (bottom up): its soft, sub-pixel
      // shifted edge then blends with almost the same picture, not with the background fill (a faint line showed)
      const hh = Math.min(h, y1 + BEND_OVERLAP_PX) - y0;
      if (hh > 0) c.drawImage(this.faceCanvas, 0, y0, w, hh, 0, y0, w, hh);
    };
    strip(this.bendBottom, h, 0);
    for (let y = this.bendBottom - BEND_STRIP_PX; y > this.bendTop - BEND_STRIP_PX; y -= BEND_STRIP_PX) {
      const y0 = Math.max(this.bendTop, y);
      const u = (y0 + BEND_STRIP_PX / 2 - this.bendTop) / Math.max(1, this.bendBottom - this.bendTop);
      strip(y0, y0 + BEND_STRIP_PX, 1 - smoothstep(0, 1, u));
    }
    strip(0, this.bendTop, 1);
    if (this.showBounds) {
      breathe();
      c.lineWidth = 2;
      const colours = { brows: "#4aa3ff", eyes: "#47d16c", mouth: "#ff6a6a" };
      for (const [region, l] of this.bounds) {
        c.strokeStyle = colours[region];
        c.strokeRect(l.x, l.y, l.w, l.h);
      }
    }
    c.setTransform(1, 0, 0, 1, 0, 0);
  }

  /** The previous head-layer sway (kept for reference; not called): a head cut-out over a body plate. */
  drawCutout(dx, dy, roll, breath) {
    const { width: w, height: h } = this.manifest;
    // body plate: the face canvas with the head cleared to background
    const p = this.plateCanvas.getContext("2d");
    p.globalCompositeOperation = "source-over";
    p.fillStyle = this.bgColor;
    p.fillRect(0, 0, w, h);
    p.drawImage(this.faceCanvas, 0, 0);
    p.globalCompositeOperation = "destination-out";
    p.drawImage(this.clearMask, 0, 0);
    p.globalCompositeOperation = "destination-over";
    p.fillRect(0, 0, w, h);
    p.globalCompositeOperation = "source-over";

    // head cut-out
    const hd = this.headCanvas.getContext("2d");
    hd.globalCompositeOperation = "source-over";
    hd.clearRect(0, 0, w, h);
    hd.drawImage(this.faceCanvas, 0, 0);
    hd.globalCompositeOperation = "destination-in";
    hd.drawImage(this.headMask, 0, 0);
    hd.globalCompositeOperation = "source-over";

    const c = this.ctx;
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.fillStyle = this.bgColor;
    c.fillRect(0, 0, w, h);
    // breathing: scale about the bottom edge (shoulders rise, head rides along)
    const s = 1 + BREATH_SCALE * breath;
    c.setTransform(1, 0, 0, s, 0, h * (1 - s));
    c.drawImage(this.plateCanvas, 0, 0);
    const [px, py] = this.pivot;
    c.translate(px + dx, py + dy);
    c.rotate(roll);
    c.translate(-px, -py);
    c.drawImage(this.headCanvas, 0, 0);
    c.setTransform(1, 0, 0, 1, 0, 0);
  }

  loop(now) {
    if (this.paused) { requestAnimationFrame(this.loop); return; }  // QA renders frames itself
    try {
      this.updateEyesAuto(now);
      const tg = this.targets(now);
      if (this.morph) this.setMorphTargets(tg, now);
      else for (const [region, [id, dur]] of Object.entries(tg)) this.fades[region].set(id, now, dur);
      this.draw(now);
      for (const fn of this.listeners) fn(this.current, now);
    } catch (error) {
      console.error(error);
    }
    requestAnimationFrame(this.loop);
  }
}
