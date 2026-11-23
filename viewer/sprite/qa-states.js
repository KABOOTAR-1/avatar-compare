/**
 * State-render QA (viewer ?qa=<port>): renders every viewer state with the REAL renderer (WebGL morph
 * or cross-fade) and posts each frame to scripts/avatar3d/sprite/qa_frame_server.py on that port.
 *
 *   states:      each expression x (its own mouth + 3 visemes) x eyes (centre / blink-mid / gaze L R up)
 *   transitions: every ordered pair of expressions, 10 evenly spaced frames over the morph
 * Animation is frozen (no breathing/sway/blinks/glances); time is synthetic, so frames are deterministic.
 * Per frame the current mesh points of every region are sent too (tearing check).
 */
const VISEMES = ["aa", "PP", "O"];
const TRANSITION_FRAMES = 10;
const TRANSITION_MS = 280;  // MORPH_EXPRESSION_MS in sprite-avatar.js
const BLINK_MS = 70;

export async function runStateQA(avatar, avatarId, port) {
  const base = `http://127.0.0.1:${port}`;
  const m = avatar.manifest;
  avatar.paused = true;
  avatar.life = false;
  avatar.autoBlink = false;
  avatar.autoGlance = false;
  const regions = avatar.morph?.regions ?? {};
  let now = 100000;
  const frames = [];

  const parts = (expr) => m.expressions[expr] ?? m.expressions.neutral;
  let cur = { brows: null, eyes: null, mouth: null };
  const setAll = (sel, dur) => {
    cur = { ...cur, ...sel };
    if (avatar.morph) {
      avatar.setMorphTargets({ brows: [cur.brows, dur], eyes: [cur.eyes, dur], mouth: [cur.mouth, dur] }, now);
    } else for (const [region, id] of Object.entries(sel)) avatar.fades[region].set(id, now, dur);
  };
  const settle = (sel) => { setAll(sel, 1); now += 50; };
  const post = async (name, meta) => {
    avatar.draw(now);
    const blob = await new Promise((r) => avatar.canvas.toBlob(r, "image/jpeg", 0.93));
    await fetch(`${base}/save?avatar=${avatarId}&name=${encodeURIComponent(name)}`, { method: "POST", body: blob });
    const mesh = {}, draws = {};
    for (const [k, r] of Object.entries(regions)) {
      if (!r.track) continue;
      const smp = r.track.sample(now);
      mesh[k] = Array.from(smp.points);
      draws[k] = smp.draws;
    }
    frames.push({ name, ...meta, mesh, draws });
  };

  const exprs = Object.keys(m.expressions);
  for (const x of exprs) {
    const p = parts(x);
    const mouths = [["own", p.mouth], ...VISEMES.filter((v) => m.visemes[v]).map((v) => [v, m.visemes[v]])];
    const eyes = [["centre", p.eyes], ["blink-mid", "blink-mid"], ["gaze-left", m.eyes.gaze.left],
      ["gaze-right", m.eyes.gaze.right], ["gaze-up", m.eyes.gaze.up]];
    for (const [mk, mouth] of mouths) {
      for (const [ek, eye] of eyes) {
        settle({ brows: p.brows, mouth, eyes: p.eyes });
        if (eye === "blink-mid") {
          setAll({ eyes: m.eyes.closed }, BLINK_MS);
          now += BLINK_MS / 2;
        } else settle({ eyes: eye });
        await post(`state__${x}__${mk}__${ek}`, { kind: "state", expression: x, mouth: mk, eyes: ek });
      }
    }
  }
  for (const a of exprs) {
    for (const b of exprs) {
      if (a === b) continue;
      settle(parts(a));
      const t0 = now;
      setAll(parts(b), TRANSITION_MS);
      for (let i = 0; i < TRANSITION_FRAMES; i++) {
        now = t0 + (TRANSITION_MS * (i + 1)) / (TRANSITION_FRAMES + 1);
        await post(`trans__${a}__${b}__${i}`, { kind: "transition", from: a, to: b, step: i });
      }
      now = t0 + TRANSITION_MS + 50;
    }
  }
  const morph = avatar.morph ? Object.fromEntries(Object.entries(regions).filter(([, r]) => r.triangles)
    .map(([k, r]) => [k, { triangles: r.triangles, base: Array.from(r.points[r.base ?? "base"]), ring: r.ring,
      states: Object.fromEntries(Object.entries(r.points).map(([s, p]) => [s, Array.from(p)])) }])) : null;
  await fetch(`${base}/meta?avatar=${avatarId}`, { method: "POST",
    body: JSON.stringify({ avatar: avatarId, renderer: avatar.renderer ?? "crossfade", frames, morph }) });
  document.title = `QA done ${frames.length}`;
  avatar.paused = false;
}
