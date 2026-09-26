// One page: pick a YouTube link or a file -> the server analyses it once -> play it here,
// and the "Now hearing" panel follows the playback using the per-slice predictions.
// Every sound has its own (corrected) score; the server sends two cut-offs with it:
// heard (>= heard_cut) and maybe (>= maybe_cut). Below that a sound is not shown.

// Helpers ($, pretty, pct, barRow, renderNow...) are in common.js, shared with the live page.

let mode = "url";
let chosenFile = null;
let requestId = 0;  // a newer Analyze click makes older answers stale
let result = null;  // the server's answer for what is loaded now
let lastWindow = -1;
const chips = new Map(); // label -> its chip in "What it can recognise"

// Whatever is playing, YouTube or a local file, is driven through these three.
let getTime = () => 0;
let seek = () => {};
let isPlaying = () => false;

// ---------- model status + the list of sounds it knows ----------
fetch("/api/info")
  .then((r) => r.json())
  .then((info) => {
    $("status").classList.add("is-ready");
    $("status-text").textContent =
      `Model ${info.model} on ${info.device.toUpperCase()}, knows ${info.classes} sounds`;
    renderGroups(info);
  })
  .catch(() => {
    $("status").classList.add("is-down");
    $("status-text").textContent = "Server not reachable";
  });

function renderGroups(info) {
  $("sounds-count").textContent =
    `${info.classes} sounds in ${info.groups.length} groups. While a clip plays, the sounds it hears light up here.`;
  $("groups").replaceChildren(
    ...info.groups.map((group) => {
      const box = document.createElement("div");
      box.className = "group";
      const title = document.createElement("h3");
      title.textContent = group.name;
      const list = document.createElement("ul");
      list.className = "chips";
      for (const label of group.labels) {
        const chip = document.createElement("li");
        chip.className = "chip";
        chip.textContent = pretty(label);
        chips.set(label, chip);
        list.append(chip);
      }
      box.append(title, list);
      return box;
    })
  );
}

// ---------- input mode ----------
document.querySelectorAll(".mode[data-mode]").forEach((btn) =>
  btn.addEventListener("click", () => {
    mode = btn.dataset.mode;
    document.querySelectorAll(".mode[data-mode]").forEach((b) => {
      b.classList.toggle("is-active", b === btn);
      b.setAttribute("aria-selected", b === btn);
    });
    document.querySelectorAll("[data-panel]").forEach((el) => (el.hidden = el.dataset.panel !== mode));
    $("error").textContent = "";
    updateButtons();
  })
);

function updateButtons() {
  $("analyze-url").disabled = !$("url").value.trim();
  $("analyze-file").disabled = !chosenFile;
}
$("url").addEventListener("input", updateButtons);

// ---------- file choice: click or drag & drop ----------
function chooseFile(file) {
  if (!file) return;
  chosenFile = file;
  $("drop").classList.add("has-file");
  $("drop-title").textContent = file.name;
  $("drop-hint").textContent = `${(file.size / 1e6).toFixed(1)} MB. Click to choose a different file.`;
  updateButtons();
}
$("file").addEventListener("change", (e) => chooseFile(e.target.files[0]));
["dragenter", "dragover"].forEach((ev) =>
  $("drop").addEventListener(ev, (e) => { e.preventDefault(); $("drop").classList.add("is-over"); })
);
["dragleave", "drop"].forEach((ev) =>
  $("drop").addEventListener(ev, (e) => { e.preventDefault(); $("drop").classList.remove("is-over"); })
);
$("drop").addEventListener("drop", (e) => chooseFile(e.dataTransfer.files[0]));

// ---------- analyze ----------
$("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const req = ++requestId;
  const url = $("url").value.trim();
  $("error").textContent = "";
  clearResult();

  // Show the player straight away, so there is something to look at while the server works.
  $("stage").hidden = false;
  if (mode === "url") {
    $("source").textContent = url;
    const id = youtubeId(url);
    if (id) showYouTube(id, url.includes("/shorts/"));
    else resetPlayers();
  } else {
    $("source").textContent = chosenFile.name;
    showFile(chosenFile);
  }
  $("listening-text").textContent =
    mode === "url" ? "Downloading the audio, then listening…" : "Listening…";
  $("stage").scrollIntoView({ behavior: "smooth", block: "start" });

  try {
    let response;
    if (mode === "file") {
      const body = new FormData();
      body.append("file", chosenFile);
      response = await fetch("/api/predict", { method: "POST", body });
    } else {
      response = await fetch("/api/predict-url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
    }
    const data = await response.json();
    if (req !== requestId) return;
    if (!response.ok) throw new Error(data.detail || `Server error ${response.status}`);

    if (mode === "url" && data.video_id) {
      if (data.video_id !== ytId) showYouTube(data.video_id, data.vertical);
      else setShape(data.vertical ? "vertical" : "wide");
    }
    showResult(data);
  } catch (err) {
    if (req !== requestId) return;
    resetPlayers();
    $("stage").hidden = true;
    $("error").textContent = err.message;
  }
});

// watch?v=ID, youtu.be/ID, /shorts/ID, /embed/ID, /live/ID. The server also
// returns the id it resolved, so a link this misses still gets its player later.
function youtubeId(url) {
  const m = url.match(/(?:youtu\.be\/|[?&]v=|\/(?:shorts|embed|live)\/)([\w-]{11})/);
  return m ? m[1] : null;
}

function setShape(shape) {
  $("stage").dataset.shape = shape; // wide | vertical | audio: decides the player's size
}

// ---------- players ----------
let yt = null;
let ytId = null;
let ytApi = null;

function youtubeApi() {
  ytApi ??= new Promise((resolve, reject) => {
    window.onYouTubeIframeAPIReady = () => resolve(window.YT);
    const script = document.createElement("script");
    script.src = "https://www.youtube.com/iframe_api";
    script.onerror = () => { ytApi = null; reject(new Error("Could not load the YouTube player.")); };
    document.head.append(script);
  });
  return ytApi;
}

function resetPlayers() {
  yt?.destroy();
  yt = null;
  ytId = null;
  $("yt-frame").hidden = true;
  $("player-error").hidden = true;
  for (const media of [$("video"), $("audio")]) {
    media.pause();
    if (media.src) URL.revokeObjectURL(media.src);
    media.removeAttribute("src");
    media.load();
  }
  $("video").hidden = true;
  $("audio-player").hidden = true;
  getTime = () => 0;
  seek = () => {};
  isPlaying = () => false;
}

function playerError(text) {
  $("player-error").textContent = text;
  $("player-error").hidden = false;
}

async function showYouTube(id, vertical) {
  resetPlayers();
  ytId = id;
  setShape(vertical ? "vertical" : "wide");
  const mount = document.createElement("div"); // the API replaces this with the iframe
  $("yt-frame").replaceChildren(mount);
  $("yt-frame").hidden = false;
  let YT;
  try {
    YT = await youtubeApi();
  } catch (err) {
    playerError(`${err.message} The results below still work.`);
    return;
  }
  if (ytId !== id) return; // another link was chosen while the API loaded
  yt = new YT.Player(mount, {
    videoId: id,
    width: "100%",
    height: "100%",
    playerVars: { playsinline: 1, rel: 0, origin: location.origin },
    events: {
      onError: (e) =>
        playerError(
          e.data === 101 || e.data === 150
            ? "The owner doesn't allow this video to play outside YouTube. The results below still work; watch it on YouTube and follow the timeline."
            : "YouTube could not play this video here. The results below still work."
        ),
    },
  });
  // Until the player is ready these methods don't exist yet, hence the ?.
  getTime = () => yt?.getCurrentTime?.() ?? 0;
  seek = (t) => { yt?.seekTo?.(t, true); };
  isPlaying = () => yt?.getPlayerState?.() === 1;
}

function showFile(file) {
  resetPlayers();
  const isVideo = file.type.startsWith("video/");
  const media = isVideo ? $("video") : $("audio");
  media.src = URL.createObjectURL(file);
  (isVideo ? $("video") : $("audio-player")).hidden = false;
  setShape(isVideo ? "wide" : "audio");
  getTime = () => media.currentTime;
  seek = (t) => { media.currentTime = t; };
  isPlaying = () => !media.paused;
}

$("video").addEventListener("loadedmetadata", (e) => {
  const v = e.target;
  if (v.videoHeight > v.videoWidth) setShape("vertical"); // a phone video
});
for (const media of [$("video"), $("audio")]) {
  media.addEventListener("error", () => {
    if (media.getAttribute("src"))
      playerError("Your browser can't play this file, but the model could still analyse it. See the results below.");
  });
}

// The big button only exists for audio; a video has its own controls.
$("play").addEventListener("click", () => {
  const audio = $("audio");
  audio.paused ? audio.play() : audio.pause();
});
for (const ev of ["play", "pause", "ended"])
  $("audio").addEventListener(ev, () => $("play").classList.toggle("is-playing", !$("audio").paused));

// ---------- results ----------
function clearResult() {
  result = null;
  lastWindow = -1;
  $("listening").hidden = false;
  $("now").hidden = true;
  $("timeline-card").hidden = true;
  $("summary-card").hidden = true;
  for (const chip of chips.values()) chip.classList.remove("is-live", "is-heard");
}

function showResult(data) {
  result = data;
  $("source").textContent = data.source;
  $("listening").hidden = true;
  $("now").hidden = false;
  const rows = Math.min(LIVE_ROWS, data.labels.length);
  $("now-bars").replaceChildren(...Array.from({ length: rows }, () => barRow()));
  renderSummary(data);
  renderTimeline(data);
  for (const row of data.overall)
    if (row.level === "heard") chips.get(row.label)?.classList.add("is-heard");
  update(true);
}

function renderSummary(data) {
  const heard = data.overall.filter((row) => row.level === "heard");
  const maybe = data.overall.filter((row) => row.level === "maybe");
  $("heard-title").textContent = `Heard (${pct0(data.heard_cut)} or more)`;
  $("maybe-title").textContent = `Maybe (${pct0(data.maybe_cut)} to ${pct0(data.heard_cut)})`;
  $("bars-heard").replaceChildren(...heard.map((row) => barRow(row.label, row.prob, row.level)));
  $("bars-maybe").replaceChildren(...maybe.map((row) => barRow(row.label, row.prob, row.level)));
  $("heard-title").hidden = $("bars-heard").hidden = heard.length === 0;
  $("maybe-title").hidden = $("bars-maybe").hidden = maybe.length === 0;
  $("nothing").hidden = data.overall.length > 0;
  $("nothing").textContent =
    `None of the ${data.labels.length} sounds it knows scored ${pct0(data.maybe_cut)} or more anywhere in this clip.`;
  $("meta").textContent =
    `${clock(data.duration)} long, ${data.windows} slice${data.windows === 1 ? "" : "s"}, ` +
    `analysed in ${data.seconds}s on ${data.device.toUpperCase()} by ${data.model}` +
    (data.calibrated ? ", corrected scores" : "") +
    (data.truncated ? ". Only the first 10 minutes were used." : "");
  $("summary-card").hidden = false;
}

// One lane per sound; a sound's segments are placed by time, so lanes overlap
// wherever several sounds were heard at once.
function renderTimeline(data) {
  // A single 5 s slice has nothing to show over time.
  $("timeline-card").hidden = data.windows < 2;
  if (data.windows < 2) return;

  $("timeline-hint").textContent =
    `One row per sound. Solid: heard (${pct0(data.heard_cut)} or more); faded: maybe ` +
    `(${pct0(data.maybe_cut)} to ${pct0(data.heard_cut)}). Rows that overlap in time are ` +
    `sounds heard together. Click anywhere to jump there.`;
  const at = (t) => `${(t / data.duration) * 100}%`;
  const describe = (lane, seg) =>
    `${clock(seg.start)}–${clock(seg.end)}  ${pretty(lane.label)}  ${seg.level}, ${pct(seg.prob)}`;

  const lanes = data.timeline.map((lane) => {
    const row = document.createElement("div");
    row.className = "lane";
    const name = document.createElement("span");
    name.className = "lane-label";
    name.textContent = pretty(lane.label);
    const track = document.createElement("div");
    track.className = "lane-track";
    for (const seg of lane.segments) {
      const el = document.createElement("div");
      el.className = `seg is-${seg.level}`;
      el.style.left = at(seg.start);
      el.style.width = at(seg.end - seg.start);
      el.title = describe(lane, seg);
      track.append(el);
    }
    row.append(name, track);
    return row;
  });
  if (lanes.length === 0) {
    const empty = document.createElement("p");
    empty.className = "hint lanes-empty";
    empty.textContent = "No sound it knows was heard anywhere in this clip.";
    lanes.push(empty);
  }
  $("lanes").replaceChildren(...lanes);
  $("axis-end").textContent = clock(data.duration);

  const rows = data.timeline
    .flatMap((lane) => lane.segments.map((seg) => ({ lane, seg })))
    .sort((x, y) => x.seg.start - y.seg.start);
  $("segments").replaceChildren(
    ...rows.map(({ lane, seg }) => {
      const tr = document.createElement("tr");
      for (const text of [clock(seg.start), clock(seg.end), pretty(lane.label), seg.level, pct(seg.prob)]) {
        const td = document.createElement("td");
        td.textContent = text;
        tr.append(td);
      }
      tr.addEventListener("click", () => seek(seg.start));
      return tr;
    })
  );
}

// Only the tracks are on the clock; a click on the sound names does nothing.
$("timeline-wrap").addEventListener("click", (e) => {
  if (!result) return;
  const box = $("track-area").getBoundingClientRect();
  const f = (e.clientX - box.left) / box.width;
  if (f >= 0 && f <= 1) seek(f * result.duration);
});

// ---------- live: follow the player ----------
// The slice whose centre is nearest to the playback time, the same rule the timeline uses.
function windowAt(t) {
  const times = result.window_times;
  let lo = 0, hi = times.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] < t) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && t - times[lo - 1] <= times[lo] - t) lo--;
  return lo;
}

function update(force = false) {
  const t = Math.min(Math.max(getTime(), 0), result.duration);
  $("playhead").style.left = `${(t / result.duration) * 100}%`;
  $("now-clock").textContent = `${clock(t)} / ${clock(result.duration)}`;
  $("now-hint").hidden = isPlaying() || t > 0;

  const k = windowAt(t);
  if (k === lastWindow && !force) return;
  lastWindow = k;

  const heard = renderNow(result.window_probs[k], result.labels, result);

  for (const chip of chips.values()) chip.classList.remove("is-live");
  for (const i of heard) chips.get(result.labels[i])?.classList.add("is-live");
}

(function tick() {
  if (result) update();
  requestAnimationFrame(tick);
})();
