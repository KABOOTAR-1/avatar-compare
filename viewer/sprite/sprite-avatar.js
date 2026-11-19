/**
 * SpriteAvatar: 2.5D talking head from layered Gemini sprites (see
 * scripts/avatar3d/sprite/build_layers.py for how the layers are cut).
 *
 * Per frame: compose face.png + brows + eyes + mouth + glasses into an offscreen canvas (each
 * region cross-fades independently, so blinks run while talking), then split it into a static
 * body plate and a head cut-out that gets a small 2D affine sway/nod around the neck, and apply
 * a breathing scale to both. Input signals are the same as a 3D avatar's: an Oculus viseme
 * timeline, an expression name, gaze, blink.
 */

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

// --- procedural motion -----------------------------------------------------------------------
const BREATH_PERIOD_S = 4.2;
const BREATH_SCALE = 0.006;      // y-scale amplitude about the bottom edge
const SWAY_DEG = 0.9;            // head roll amplitude
const SWAY_PX = 1.5;             // head drift amplitude (image px)
const TALK_NOD_DEG = 0.6;        // extra nodding while talking
const NECK_FEATHER_PX = 14;      // head/body blend band around the neck cut (kept below the jaw)
const NECK_CUT_BELOW_CHIN = 0.53; // cut line: chin + this x (pivot - chin); below the dropped jaw of "aa", above the collar
const BG_MATTE_LO = 6;           // colour distance from the flat background that counts as figure
const HEAD_MATTE_MARGIN_PX = 6;  // matte grown into pure background so its edge is invisible

const REGIONS = ["brows", "eyes", "mouth"];

const loadImage = (src) => new Promise((resolve, reject) => {
  const img = new Image();
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
    this.manifest = await (await fetch(`${this.root}manifest.json`)).json();
    const m = this.manifest;
    const files = { face: m.base };
    for (const [region, layers] of Object.entries(m.layers)) {
      for (const [id, layer] of Object.entries(layers)) files[`${region}/${id}`] = layer.file;
    }
    const entries = await Promise.all(Object.entries(files).map(async ([k, f]) => [k, await loadImage(this.root + f)]));
    this.images = Object.fromEntries(entries);
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
    const grown = dilate(fg, w, h, HEAD_MATTE_MARGIN_PX);
    const chinY = anchors.chin[1];
    const cutY = chinY + NECK_CUT_BELOW_CHIN * (anchors.neck_pivot[1] - chinY);
    this.pivot = [anchors.neck_pivot[0], cutY];
    const head = g.createImageData(w, h);
    const clear = g.createImageData(w, h);
    for (let y = 0; y < h; y++) {
      const v = 1 - smoothstep(cutY - NECK_FEATHER_PX, cutY + NECK_FEATHER_PX, y);
      const above = y < cutY - NECK_FEATHER_PX;
      for (let x = 0; x < w; x++) {
        const k = y * w + x;
        head.data[k * 4 + 3] = grown[k] ? Math.round(255 * v) : 0;
        clear.data[k * 4 + 3] = grown[k] && above ? 255 : 0;
      }
    }
    const toCanvas = (img) => {
      const k = Object.assign(document.createElement("canvas"), { width: w, height: h });
      k.getContext("2d").putImageData(img, 0, 0);
      return k;
    };
    this.headMask = toCanvas(head);
    this.clearMask = toCanvas(clear);
  }

  // --- public controls ------------------------------------------------------------------------
  setExpression(name) { this.expression = name in this.manifest.expressions ? name : "neutral"; }
  setGaze(dir) { this.gaze = dir; }
  previewViseme(v) { this.mouthOverride = v; }
  blink(now = performance.now()) { this.blinkQueue = this.makeBlink(now, false); }
  onUpdate(fn) { this.listeners.add(fn); }

  /** Play a speech clip: {audio url, visemes:[{start,end,viseme,phone}], words, expression}. */
  async say(clip) {
    this.stopSpeech();
    const audio = new Audio(this.root + clip.audio);
    this.speech = { clip, audio };
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
  }

  // --- state machine --------------------------------------------------------------------------
  makeBlink(now, allowDouble = true) {
    const steps = [];
    let t = now;
    const reps = allowDouble && Math.random() < DOUBLE_BLINK_CHANCE ? 2 : 1;
    for (let r = 0; r < reps; r++) {
      for (const [state, ms] of BLINK_SEQUENCE) { steps.push({ state, at: t }); t += ms; }
      steps.push({ state: null, at: t });
      t += 90;
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
    const t = this.speech.audio.currentTime + VISEME_LEAD_S;
    const { visemes, words } = this.speech.clip;
    const cue = visemes.find((c) => t >= c.start && t < c.end) ?? visemes[visemes.length - 1];
    const word = words?.find((w) => t >= w.start && t < w.end);
    return { viseme: cue.viseme, phone: cue.phone, word: word?.word ?? "" };
  }

  /** Decide each region's target layer id. */
  targets(now) {
    const m = this.manifest;
    const expr = m.expressions[this.expression] ?? m.expressions.neutral;
    const speaking = this.currentViseme();
    let mouth = expr.mouth;
    let viseme = "sil";
    if (speaking) { viseme = speaking.viseme; mouth = m.visemes[viseme] ?? null; }
    if (this.mouthOverride) { viseme = this.mouthOverride; mouth = m.visemes[viseme] ?? null; }
    this.current = { viseme, phone: speaking?.phone ?? "", word: speaking?.word ?? "" };

    // eyes: blink > gaze > expression. Expressions with their own closed eyes (laugh) do not blink.
    const eyesClosedByExpr = this.expression === "laugh";
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
    const mouthFade = speaking || this.mouthOverride ? this.visemeFadeMs : EXPRESSION_FADE_MS;
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
    this.composeFace(now);
    const t = now / 1000;
    const talking = this.speech ? 1 : 0;
    // smooth pseudo-noise from incommensurate sines
    const n1 = Math.sin(t * 0.73) * 0.6 + Math.sin(t * 1.31 + 1.7) * 0.4;
    const n2 = Math.sin(t * 0.57 + 0.4) * 0.6 + Math.sin(t * 1.13 + 2.9) * 0.4;
    const n3 = Math.sin(t * 2.3) * 0.5 + Math.sin(t * 3.7 + 0.8) * 0.5;
    const life = this.life ? 1 : 0;
    const breath = life * Math.sin((2 * Math.PI * t) / BREATH_PERIOD_S);
    const roll = life * (n1 * SWAY_DEG + talking * n3 * TALK_NOD_DEG * 0.5) * (Math.PI / 180);
    const dx = life * n2 * SWAY_PX;
    const dy = life * (n1 * SWAY_PX * 0.4 + talking * Math.abs(n3) * 1.5);

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
    if (this.showBounds) {
      c.lineWidth = 2;
      const colours = { brows: "#4aa3ff", eyes: "#47d16c", mouth: "#ff6a6a" };
      for (const [region, l] of this.bounds) {
        c.strokeStyle = colours[region];
        c.strokeRect(l.x, l.y, l.w, l.h);
      }
    }
    c.setTransform(1, 0, 0, 1, 0, 0);
  }

  loop(now) {
    try {
      this.updateEyesAuto(now);
      for (const [region, [id, dur]] of Object.entries(this.targets(now))) this.fades[region].set(id, now, dur);
      this.draw(now);
      for (const fn of this.listeners) fn(this.current, now);
    } catch (error) {
      console.error(error);
    }
    requestAnimationFrame(this.loop);
  }
}
