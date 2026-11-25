/**
 * UI wiring for the 2.5D sprite avatar demo. Served from avatar-eval/ (python -m http.server
 * 8765 in avatar-eval/): http://127.0.0.1:8765/viewer/sprite/index.html[?avatar=arvind]
 */
import { SpriteAvatar, VISEME_FADE_MS } from "./sprite-avatar.js";

const DEFAULT_AVATAR = "arvind";
const VISEME_PREVIEW_MS = 700;
const EXPRESSION_LABELS = { neutral: "Neutral", smile: "Smile", laugh: "Laugh", sad: "Sad", surprised: "Surprised", thinking: "Thinking" };

const params = new URLSearchParams(location.search);
// keep only id characters: links pasted from chat often drag a trailing ")" or "." along
const avatarId = (params.get("avatar") ?? DEFAULT_AVATAR).replace(/[^A-Za-z0-9_-]/g, "") || DEFAULT_AVATAR;
const root = `../../out/sprite/${avatarId}/`;
const $ = (id) => document.getElementById(id);

function button(label, onClick, title = "") {
  const b = document.createElement("button");
  b.textContent = label;
  b.title = title;
  b.addEventListener("click", onClick);
  return b;
}

function radioGroup(container, items, onSelect, initial) {
  const buttons = new Map();
  const select = (key) => {
    for (const [k, b] of buttons) b.classList.toggle("on", k === key);
    onSelect(key);
  };
  for (const [key, label] of items) {
    const b = button(label, () => select(key));
    buttons.set(key, b);
    container.append(b);
  }
  if (initial !== undefined) select(initial);
  return select;
}

/** Local work-in-progress badge (this viewer is not the published one). */
function wipBadge() {
  const b = document.createElement("div");
  b.textContent = "WIP · local";
  b.title = "Work in progress: local viewer with the morph renderer, not yet published";
  Object.assign(b.style, { position: "fixed", top: "10px", right: "10px", zIndex: 1000, padding: "4px 10px",
    borderRadius: "6px", background: "#c2410c", color: "#fff", font: "600 13px system-ui, sans-serif" });
  document.body.append(b);
}

async function main() {
  if (["127.0.0.1", "localhost"].includes(location.hostname)) wipBadge();
  const avatar = new SpriteAvatar($("stage"), root);
  if (params.get("qa")) avatar.bust = `?t=${Date.now()}`;
  try {
    await avatar.load();
  } catch (error) {
    $("status").textContent = `Could not load ${root}: ${error.message}`;
    return;
  }
  const m = avatar.manifest;
  // local runs have raw/base.png; a published web/ copy has raw/base.webp
  $("reference").onerror = () => { $("reference").onerror = null; $("reference").src = `${root}raw/base.webp`; };
  $("reference").src = `${root}raw/base.png`;
  $("status").textContent = `${avatarId}: ${Object.values(m.layers).reduce((n, l) => n + Object.keys(l).length, 0)} layers`
    + (avatar.morph ? ` · morph (${avatar.renderer})` : " · cross-fade");
  if (!m.morph) {  // the pipeline writes the morph data near its end: before that this is a half-built avatar
    $("status").textContent += " · STILL BUILDING (preview only: no morphing, no speech yet; reload when it finishes)";
    $("status").style.color = "#f0a040";
  }

  let chosenExpression = "neutral";  // the user's choice; a line's own expression only lasts while it is spoken
  const selectExpression = radioGroup($("expressions"), Object.keys(m.expressions).map((k) => [k, EXPRESSION_LABELS[k] ?? k]),
    (k) => { chosenExpression = k; avatar.setExpression(k); }, "neutral");
  const restoreExpression = () => avatar.setExpression(chosenExpression);

  radioGroup($("gaze"), [[null, "Centre"], ...Object.keys(m.eyes.gaze).map((k) => [k, k[0].toUpperCase() + k.slice(1)])],
    (k) => avatar.setGaze(k), null);
  $("gaze").append(button("Blink", () => avatar.blink()));

  let previewTimer = 0;
  for (const v of Object.keys(m.visemes)) {
    const target = m.visemes[v];
    // "sil" (silence) IS the resting mouth: labelled so pressing it does not look like a broken button
    const b = button(target === null ? `${v} (rest)` : v, () => {
      avatar.previewViseme(v);
      clearTimeout(previewTimer);
      previewTimer = setTimeout(() => avatar.previewViseme(null), VISEME_PREVIEW_MS);
    }, target === null ? "base mouth" : target === v ? "" : `reuses ${target}`);
    $("visemes").append(b);
  }
  // Azure 22-viseme set (optional manifest key): one button per id; 9-11 preview their glide
  if (m.visemes_azure) {
    const label = document.createElement("span");
    label.textContent = " Azure:";
    $("visemes").append(document.createElement("br"), label);
    for (const [id, target] of Object.entries(m.visemes_azure)) {
      const title = target === null ? "base mouth" : typeof target === "string" ? target : `glide ${target.glide.join(" -> ")}`;
      $("visemes").append(button(id, () => {
        avatar.previewViseme(`az:${id}`);
        clearTimeout(previewTimer);
        previewTimer = setTimeout(() => avatar.previewViseme(null), VISEME_PREVIEW_MS);
      }, title));
    }
  }

  // speech clips
  let index = [];
  try {
    index = await (await fetch(`${root}speech/index.json`)).json();
  } catch (error) {
    console.warn("no speech clips", error);
  }
  const clips = {};
  for (const item of index) {
    const b = button(`Talk: ${item.id}`, async () => {
      try {
        clips[item.id] ??= await (await fetch(root + item.file)).json();
      } catch (error) {
        $("caption").textContent = `Could not load ${item.file}: ${error.message}`;
        return;
      }
      const clip = clips[item.id];
      $("caption").textContent = clip.text;
      if ($("withExpr").checked) avatar.setExpression(clip.expression in m.expressions ? clip.expression : chosenExpression);
      await avatar.say(clip);
      avatar.speech?.audio.addEventListener("ended", restoreExpression);  // back to the chosen expression afterwards
    }, item.text);
    $("talk").append(b);
  }
  $("talk").append(button("Stop", () => { avatar.stopSpeech(); restoreExpression(); }));
  // ?qa=<port>: render every viewer state + expression transitions and send them to qa_frame_server.py
  if (params.get("qa")) (await import("./qa-states.js")).runStateQA(avatar, avatarId, Number(params.get("qa")));

  $("life").addEventListener("change", (e) => { avatar.life = e.target.checked; });
  $("autoBlink").addEventListener("change", (e) => { avatar.autoBlink = e.target.checked; });
  $("autoGlance").addEventListener("change", (e) => { avatar.autoGlance = e.target.checked; });
  $("bounds").addEventListener("change", (e) => { avatar.showBounds = e.target.checked; });
  $("fade").value = VISEME_FADE_MS;
  $("fadeVal").textContent = `${VISEME_FADE_MS} ms`;
  $("fade").addEventListener("input", (e) => {
    avatar.visemeFadeMs = Number(e.target.value);
    $("fadeVal").textContent = `${e.target.value} ms`;
  });

  let frames = 0;
  let last = performance.now();
  avatar.onUpdate((cur, now) => {
    frames++;
    if (now - last > 500) {
      $("fps").textContent = `${Math.round((frames * 1000) / (now - last))} fps`;
      frames = 0;
      last = now;
    }
    $("readout").textContent = `viseme ${cur.viseme}${cur.phone ? ` (${cur.phone})` : ""}${cur.word ? ` · “${cur.word}”` : ""}`;
  });
}

main();
