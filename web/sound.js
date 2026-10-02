// One sound's page (sound.html?label=dog): where its clips come from, how well
// the model finds it on the test fold, and three typical training clips as
// log-mel pictures you can play. Everything comes from /api/sound/<label>.

const label = new URLSearchParams(location.search).get("label") || "";

fetch(`/api/sound/${encodeURIComponent(label)}`)
  .then(async (r) => {
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || `Server error ${r.status}`);
    render(data);
  })
  .catch((err) => {
    $("name").textContent = pretty(label) || "Unknown sound";
    $("error").textContent = err.message === "Failed to fetch" ? "Server not reachable" : err.message;
  });

function render(d) {
  const name = pretty(d.label);
  document.title = `Sound tagger: ${name}`;
  $("name").textContent = name;
  $("group").textContent = d.group;
  for (const [id, other] of [["prev", d.previous], ["next", d.next]]) {
    $(id).href = `sound.html?label=${encodeURIComponent(other)}`;
    $(id).querySelector("span").textContent = pretty(other);
  }

  const facts = [];
  facts.push(["Where the clips come from", d.esc50
    ? "ESC-50: 5-second clips cut from longer recordings on Freesound, 40 per sound."
    : `Added by us: the clips in data/custom/${d.label}/.`]);
  if (d.counts) {
    const c = d.counts;
    facts.push(["Clips", `${c.train + c.test + c.practice}, cut from ${c.recordings} original recording${c.recordings === 1 ? "" : "s"}` +
      (d.esc50 ? "" : " (the part of each file name before __; for speech, the speaker)")]);
    facts.push(["Used for", `${c.train} trained on; ${c.test} held out for the test (fold ${d.test_fold})` +
      (c.practice ? `; ${c.practice} for the practice exam (fold ${d.practice_fold})` : "") +
      ". Clips from one recording always stay in the same fold."]);
  }
  if (d.ap !== null)
    facts.push(["How well it finds it", `AP ${d.ap.toFixed(2)} on the fold-${d.test_fold} test mixes, ` +
      `number ${d.ap_rank} of ${d.ap_of} sounds. 1.00 would mean every test mix containing ${name} ` +
      `scored higher for it than every mix without it.`]);
  if (d.panns)
    facts.push(["vs PANNs CNN14", `Of its single fold-${d.test_fold} test clips, ours named ` +
      `${pct0(d.panns.ours)} right, PANNs (pretrained on 2 M clips, not fine-tuned) ${pct0(d.panns.panns)}.`]);
  facts.push(["Model", d.model]);
  $("facts").replaceChildren(...facts.flatMap(([term, text]) => {
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    dd.textContent = text;
    return [dt, dd];
  }));
  $("facts-card").hidden = false;

  $("examples-hint").textContent = d.examples.length
    ? `The ${d.examples.length} training clips closest to the average of all ${d.counts.train} training ` +
      `clips of ${name}, as the model sees them (its 256-number summary), each from a different ` +
      `recording. Same colour scale as everywhere on the site.`
    : "No examples: the server has no data/ folder. Run scripts/download_data.py and restart it.";
  $("examples").replaceChildren(...d.examples.map((ex) => {
    const fig = document.createElement("figure");
    fig.className = "spec";
    fig.innerHTML = `<figcaption></figcaption><img alt=""><audio controls preload="none"></audio>`;
    fig.querySelector("figcaption").textContent = ex.name;
    fig.querySelector("img").src = `/api/clip/${ex.id}.png`;
    fig.querySelector("img").alt = `Log-mel spectrogram of ${ex.name}`;
    fig.querySelector("audio").src = `/api/clip/${ex.id}.wav`;
    return fig;
  }));
  $("examples-card").hidden = false;
}
