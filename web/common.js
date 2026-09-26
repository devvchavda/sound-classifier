// Shared by both pages (index.html: files and YouTube; live.html: camera + mic):
// small helpers, the probability bars and the "Now hearing" panel.

const $ = (id) => document.getElementById(id);
const pretty = (label) => label.replaceAll("_", " ");
const pct = (p) => `${(p * 100).toFixed(1)}%`;
const pct0 = (p) => `${Math.round(p * 100)}%`;
const clock = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const LIVE_ROWS = 5;

// "heard", "maybe" or "" (not shown). `cuts` is anything with heard_cut and maybe_cut
// (the server's /api/info or one analysis result).
function levelOf(p, cuts) {
  return p >= cuts.heard_cut ? "heard" : p >= cuts.maybe_cut ? "maybe" : "";
}

function barRow(label = "", prob = 0, level = "") {
  const li = document.createElement("li");
  li.innerHTML = `<span class="bar-label"></span>
    <span class="bar-track"><span class="bar-fill"></span></span>
    <span class="bar-value"></span>`;
  setBar(li, label, prob, level);
  return li;
}

// Solid lime = heard, faded lime = maybe, grey = below the maybe cut-off.
function setBar(li, label, prob, level) {
  li.className = "bar" + (level ? ` is-${level}` : "");
  li.querySelector(".bar-label").textContent = pretty(label);
  li.querySelector(".bar-fill").style.width = `${Math.max(prob * 100, 0.5)}%`;
  li.querySelector(".bar-value").textContent = pct(prob);
}

// Fills #now-label, #now-sub and the rows of #now-bars from one slice's scores.
// Returns the indices of the sounds heard, so a page can light them up elsewhere.
function renderNow(probs, labels, cuts) {
  const order = probs.map((_, i) => i).sort((a, b) => probs[b] - probs[a]);
  const heard = order.filter((i) => levelOf(probs[i], cuts) === "heard");
  const maybe = order.filter((i) => levelOf(probs[i], cuts) === "maybe");

  // Big line: every sound heard now; else the best "maybe"; else nothing.
  const label = $("now-label");
  if (heard.length) label.textContent = heard.map((i) => pretty(labels[i])).join(" + ");
  else if (maybe.length) label.textContent = `maybe ${pretty(labels[maybe[0]])}`;
  else label.textContent = "Nothing it recognises";
  label.classList.toggle("is-unsure", heard.length === 0);
  label.classList.toggle("is-many", heard.length > 1 || heard.length + maybe.length === 0);

  const counts = [heard.length && `${heard.length} heard`, maybe.length && `${maybe.length} maybe`]
    .filter(Boolean).join(", ");
  $("now-sub").textContent = counts
    ? `${counts} in this slice`
    : `No sound scored ${pct0(cuts.maybe_cut)} or more in this slice.`;
  [...$("now-bars").children].forEach((li, i) =>
    setBar(li, labels[order[i]], probs[order[i]], levelOf(probs[order[i]], cuts)));
  return heard;
}
