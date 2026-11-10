const $ = id => document.getElementById(id);
const EXAMPLE = ["VVYPW", "LVVYPW", "VVYPWTQ", "TSKY", "LFP", "IA", "AL", "VLDAFSDGLK"].join("\n");
let LAST = [];
let SORT = { key: null, dir: 1 };

function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function loadInfo() {
  try {
    const r = await fetch("/api/info");
    const d = await r.json();
    $("title").textContent = d.title;
    $("subtitle").textContent =
      `Paste peptide sequences and the server predicts ` +
      `${d.class1_meaning.toLowerCase()} versus ${d.class0_meaning.toLowerCase()} (${d.activity_definition}).`;
    // The citation row appears only once a real citation has been set in app.py.
    const placeholder = "Add your paper citation / DOI here.";
    const citeRow = (d.citation && d.citation !== placeholder)
      ? `<tr><td>Citation</td><td>${esc(d.citation)}</td></tr>` : "";
    $("about").innerHTML = `<table>
      <tr><td>Model</td><td>${esc(d.model_display || d.model)}</td></tr>
      <tr><td>Positive class</td><td>${esc(d.class1_meaning)} &mdash; ${esc(d.activity_definition)}</td></tr>
      <tr><td>Decision threshold</td><td>probability &ge; ${d.threshold}</td></tr>
      <tr><td>Accepted input</td><td>20 standard amino acids, length ${d.min_length}&ndash;${d.max_length}, up to ${d.max_sequences} sequences per run</td></tr>
      ${citeRow}</table>`;
    const cPlace = "Add a contact name and e-mail here.";
    const contact = (d.contact && d.contact !== cPlace) ? ` \u00b7 ${d.contact}` : "";
    $("footer").textContent = `${d.title} v${d.version}${contact}`;
  } catch { $("about").textContent = "Could not reach the server."; }
}

function rowsHTML(rows) {
  return rows.map(r => {
    let pred, prob = "";
    if (r.status === "rejected") {
      pred = '<span class="pill rejected">rejected</span>';
    } else {
      const act = r.label === 1;
      pred = `<span class="pill ${act ? "active" : "inactive"}">${act ? "Active" : "Inactive"}</span>`;
      const pct = (r.probability * 100).toFixed(1);
      prob = `${r.probability.toFixed(3)}
        <div class="bar"><i style="width:${pct}%;background:${act ? "#E8871A" : "#3B0F87"}"></i></div>`;
    }
    const dom = r.in_domain === null ? ""
      : r.in_domain ? `<span class="dom-in">inside</span> <span class="note">(${r.ad_distance})</span>`
                    : `<span class="dom-out">outside</span> <span class="note">(${r.ad_distance})</span>`;
    return `<tr>
      <td class="num">${r.id}</td><td>${esc(r.name)}</td>
      <td class="seq">${esc(r.sequence)}</td><td class="num">${r.length}</td>
      <td class="num">${prob}</td><td>${pred}</td><td>${dom}</td>
      <td class="note">${esc(r.note || "")}</td></tr>`;
  }).join("");
}

function sortedRows() {
  if (!SORT.key) return LAST;
  const { key, dir, type } = SORT;
  return LAST.slice().sort((a, b) => {
    let x = a[key], y = b[key];
    // rows without a value (rejected peptides) always sink to the bottom
    const xn = x === null || x === undefined, yn = y === null || y === undefined;
    if (xn && yn) return a.id - b.id;
    if (xn) return 1;
    if (yn) return -1;
    if (type === "num") return (x - y) * dir || a.id - b.id;
    return String(x).localeCompare(String(y)) * dir || a.id - b.id;
  });
}

function paint() {
  $("results").querySelector("tbody").innerHTML = rowsHTML(sortedRows());
  document.querySelectorAll("th.sortable").forEach(th => {
    th.classList.remove("asc", "desc");
    if (th.dataset.key === SORT.key) th.classList.add(SORT.dir === 1 ? "asc" : "desc");
  });
}

function render(data) {
  LAST = data.results;
  // most useful default for screening: strongest predictions first
  SORT = { key: "probability", dir: -1, type: "num" };
  const s = data.summary;
  $("summary").innerHTML =
    `<b>${s.total}</b> sequence(s) &middot; <b>${s.class1}</b> predicted active &middot; ` +
    `<b>${s.class0}</b> predicted inactive &middot; <b>${s.in_domain}</b> inside the applicability domain` +
    (s.rejected ? ` &middot; <b>${s.rejected}</b> rejected` : "");
  paint();
  $("resultCard").classList.remove("hidden");
}

async function run() {
  const text = $("seqbox").value.trim();
  if (!text) { $("status").textContent = "Paste at least one sequence."; return; }
  $("run").disabled = true; $("status").textContent = "Computing descriptors and predicting…";
  try {
    const r = await fetch("/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text })
    });
    const d = await r.json();
    if (!r.ok) { $("status").textContent = d.detail || "Request failed."; return; }
    render(d); $("status").textContent = "Done.";
  } catch (e) {
    $("status").textContent = "Could not reach the server.";
  } finally { $("run").disabled = false; }
}

function downloadCSV() {
  if (!LAST.length) return;
  const head = ["id", "name", "sequence", "length", "probability", "predicted_label",
                "in_applicability_domain", "ad_distance", "status", "note"];
  const esc2 = v => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const rows = sortedRows().map(r => [r.id, r.name, r.sequence, r.length, r.probability, r.label,
                              r.in_domain, r.ad_distance, r.status, r.note].map(esc2).join(","));
  const blob = new Blob([head.join(",") + "\n" + rows.join("\n")], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "peptide_predictions.csv"; a.click();
  URL.revokeObjectURL(a.href);
}

document.querySelectorAll("th.sortable").forEach(th => {
  th.onclick = () => {
    const key = th.dataset.key, type = th.dataset.type;
    SORT = (SORT.key === key) ? { key, dir: -SORT.dir, type } : { key, dir: type === "num" ? -1 : 1, type };
    paint();
  };
});

$("run").onclick = run;
$("example").onclick = () => { $("seqbox").value = EXAMPLE; $("status").textContent = ""; };
$("clear").onclick = () => { $("seqbox").value = ""; $("resultCard").classList.add("hidden"); $("status").textContent = ""; };
$("download").onclick = downloadCSV;
$("upload").onclick = () => $("file").click();
$("file").onchange = e => {
  const f = e.target.files[0]; if (!f) return;
  const fr = new FileReader();
  fr.onload = () => { $("seqbox").value = fr.result; $("status").textContent = `Loaded ${f.name}.`; };
  fr.readAsText(f);
};
$("seqbox").addEventListener("keydown", e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) run(); });
loadInfo();
