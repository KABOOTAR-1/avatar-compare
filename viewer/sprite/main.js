/**
 * UI wiring for the 2.5D sprite avatar demo. Served from avatar-eval/ (python -m http.server
 * 8765 in avatar-eval/): http://127.0.0.1:8765/viewer/sprite/index.html[?avatar=arvind]
 */
import { SpriteAvatar, VISEME_FADE_MS } from "./sprite-avatar.js";

const DEFAULT_AVATAR = "arvind";
const VISEME_PREVIEW_MS = 700;
const EXPRESSION_LABELS = { neutral: "Neutral", smile: "Smile", laugh: "Laugh", sad: "Sad", surprised: "Surprised", thinking: "Thinking" };

const params = new URLSearchParams(location.search);
const avatarId = params.get("avatar") ?? DEFAULT_AVATAR;
const root = `../../out/sprite/${avatarId}/`; // same site: dozens of small layer images, which a CDN/raw host throttles;
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

async function main() {
  const avatar = new SpriteAvatar($("stage"), root);
  try {
    await avatar.load();
  } catch (error) {
    $("status").textContent = `Could not load ${root}: ${error.message}`;
    return;
  }
  const m = avatar.manifest;
  $("reference").src = `${root}raw/base.webp`;
  $("status").textContent = `${avatarId}: ${Object.values(m.layers).reduce((n, l) => n + Object.keys(l).length, 0)} layers`;

  const selectExpression = radioGroup($("expressions"),
    Object.keys(m.expressions).map((k) => [k, EXPRESSION_LABELS[k] ?? k]), (k) => avatar.setExpression(k), "neutral");

  radioGroup($("gaze"), [[null, "Centre"], ...Object.keys(m.eyes.gaze).map((k) => [k, k[0].toUpperCase() + k.slice(1)])],
    (k) => avatar.setGaze(k), null);
  $("gaze").append(button("Blink", () => avatar.blink()));

  let previewTimer = 0;
  for (const v of Object.keys(m.visemes)) {
    const target = m.visemes[v];
    const b = button(v, () => {
      avatar.previewViseme(v);
      clearTimeout(previewTimer);
      previewTimer = setTimeout(() => avatar.previewViseme(null), VISEME_PREVIEW_MS);
    }, target === null ? "base mouth" : target === v ? "" : `reuses ${target}`);
    $("visemes").append(b);
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
      if ($("withExpr").checked) selectExpression(clip.expression in m.expressions ? clip.expression : "neutral");
      avatar.say(clip);
    }, item.text);
    $("talk").append(b);
  }
  $("talk").append(button("Stop", () => avatar.stopSpeech()));

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
