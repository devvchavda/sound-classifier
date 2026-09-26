// Live page: camera + microphone -> every 0.5 s the new microphone samples go to the
// server over a WebSocket (/ws/live) -> it answers with the scores of the last 5 s.
// The camera is only shown here; only the sound is sent.

const SEND_EVERY_MS = 500;
const HISTORY_S = 30; // the timeline shows the last 30 s
const STEP_S = SEND_EVERY_MS / 1000;

let info = null;     // /api/info: labels + the heard/maybe cut-offs
let session = null;  // { stream, ctx, ws, timer, t0 } while running
let history = [];    // [{ t, probs }], the last HISTORY_S seconds of answers

fetch("/api/info")
  .then((r) => r.json())
  .then((data) => {
    info = data;
    $("status").classList.add("is-ready");
    $("status-text").textContent =
      `Model ${info.model} on ${info.device.toUpperCase()}, knows ${info.classes} sounds`;
    $("now-bars").replaceChildren(...Array.from({ length: Math.min(LIVE_ROWS, info.labels.length) }, () => barRow()));
    $("timeline-hint").textContent =
      `One row per sound. Solid: heard (${pct0(info.heard_cut)} or more); faded: maybe ` +
      `(${pct0(info.maybe_cut)} to ${pct0(info.heard_cut)}). Newest on the right.`;
    $("start").disabled = false;
  })
  .catch(() => {
    $("status").classList.add("is-down");
    $("status-text").textContent = "Server not reachable";
  });

$("start").addEventListener("click", () => (session ? stop() : start()));

async function start() {
  $("error").textContent = "";
  $("start").disabled = true;

  // Browsers clean up the mic for video calls: they remove "background noise" and
  // keep the voice. For us the dog, the keyboard and the clapping ARE the sound,
  // so all three clean-ups are switched off.
  const mic = { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 };
  let stream;
  let withCamera = true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: mic });
  } catch {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: mic }); // no camera: sound only
      withCamera = false;
    } catch (err) {
      $("error").textContent =
        `Could not open the microphone (${err.name}). Allow it in the browser's address bar and try again.`;
      $("start").disabled = false;
      return;
    }
  }

  // Microphone -> a tiny audio-thread helper that hands us the raw samples.
  // It must be connected through to the speakers to keep running, so it goes via a mute.
  const ctx = new AudioContext();
  await ctx.audioWorklet.addModule("mic-worklet.js");
  const tap = new AudioWorkletNode(ctx, "mic-tap");
  const mute = ctx.createGain();
  mute.gain.value = 0;
  ctx.createMediaStreamSource(stream).connect(tap).connect(mute).connect(ctx.destination);
  let pending = [];
  tap.port.onmessage = (e) => pending.push(e.data);

  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/live`);
  ws.onopen = () => ws.send(JSON.stringify({ sample_rate: ctx.sampleRate }));
  ws.onmessage = (e) => onScores(JSON.parse(e.data));
  ws.onclose = () => {
    if (session?.ws !== ws) return; // we closed it ourselves
    stop();
    $("error").textContent = "Lost the connection to the server. Is it still running?";
  };

  // Every 0.5 s: everything the mic gave since last time, as float32 samples.
  const timer = setInterval(() => {
    if (ws.readyState !== WebSocket.OPEN || pending.length === 0) return;
    const chunk = new Float32Array(pending.reduce((n, a) => n + a.length, 0));
    let at = 0;
    for (const a of pending) { chunk.set(a, at); at += a.length; }
    pending = [];
    showMicLevel(chunk);
    ws.send(chunk.buffer);
  }, SEND_EVERY_MS);

  session = { stream, ctx, ws, timer, t0: performance.now() };
  history = [];
  $("camera").srcObject = stream;
  $("camera").hidden = !withCamera;
  $("no-camera").hidden = withCamera;
  $("stage").hidden = false;
  $("start").textContent = "Stop";
  $("start").disabled = false;
  $("now-label").textContent = "Listening…";
  $("stage").scrollIntoView({ behavior: "smooth", block: "start" });
}

function stop() {
  if (!session) return;
  const { stream, ctx, ws, timer } = session;
  session = null; // first, so ws.onclose knows this was on purpose
  clearInterval(timer);
  ws.close();
  stream.getTracks().forEach((track) => track.stop()); // camera light goes off
  ctx.close();
  $("camera").srcObject = null;
  $("mic-fill").style.width = "0%";
  $("now-clock").textContent = "stopped";
  $("start").textContent = "Start camera + mic";
}

// Loudness of the last 0.5 s, so you can see the mic is picking something up.
function showMicLevel(chunk) {
  let sum = 0;
  for (const x of chunk) sum += x * x;
  const db = 20 * Math.log10(Math.sqrt(sum / chunk.length) + 1e-9); // -60 dB quiet ... 0 dB loudest
  $("mic-fill").style.width = `${Math.min(Math.max((db + 60) / 60, 0), 1) * 100}%`;
}

function onScores({ probs, filled }) {
  if (!session) return;
  const t = (performance.now() - session.t0) / 1000;
  history.push({ t, probs });
  history = history.filter((h) => h.t > t - HISTORY_S);

  renderNow(probs, info.labels, info);
  $("now-clock").textContent =
    filled < 5 ? `window filling: ${filled.toFixed(1)} of 5 s` : `live ${clock(t)}, last 5 s`;
  renderTimeline(t);
}

// One lane per sound heard or maybe in the last 30 s, in the order they first
// appeared (so lanes don't jump around). Each answer is a 0.5 s block at its time.
function renderTimeline(now) {
  const first = new Map(); // sound index -> first time it showed up
  for (const h of history)
    h.probs.forEach((p, i) => { if (levelOf(p, info) && !first.has(i)) first.set(i, h.t); });
  const sounds = [...first.keys()].sort((a, b) => first.get(a) - first.get(b));

  const lanes = sounds.map((i) => {
    const row = document.createElement("div");
    row.className = "lane";
    const name = document.createElement("span");
    name.className = "lane-label";
    name.textContent = pretty(info.labels[i]);
    const track = document.createElement("div");
    track.className = "lane-track";
    for (const h of history) {
      const level = levelOf(h.probs[i], info);
      if (!level) continue;
      const el = document.createElement("div");
      el.className = `seg is-${level}`;
      el.style.left = `${((h.t - STEP_S - (now - HISTORY_S)) / HISTORY_S) * 100}%`;
      el.style.width = `${(STEP_S / HISTORY_S) * 100}%`;
      el.title = `${pretty(info.labels[i])}  ${level}, ${pct(h.probs[i])}`;
      track.append(el);
    }
    row.append(name, track);
    return row;
  });
  if (lanes.length === 0) {
    const empty = document.createElement("p");
    empty.className = "hint lanes-empty";
    empty.textContent = "Nothing it recognises yet. Make a sound!";
    lanes.push(empty);
  }
  $("lanes").replaceChildren(...lanes);
}

// Leaving the page turns the camera and mic off.
window.addEventListener("pagehide", stop);
