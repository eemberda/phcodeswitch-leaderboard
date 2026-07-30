import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  {css}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>{title}</h1>
    <p class="subtitle">Standardized benchmark for Cebuano-English code-switched ASR models</p>
    <p class="timestamp">Generated: {timestamp} | Entries: {entry_count}</p>
  </header>

  <div class="filters">
    <label>Benchmark:
      <select id="benchmark-select" onchange="filterTable()">
        {benchmark_options}
      </select>
    </label>
    <label>Sort by:
      <select id="sort-select" onchange="filterTable()">
        <option value="corpus_wer">WER (asc)</option>
        <option value="corpus_cer">CER (asc)</option>
        <option value="mean_case_wer">Mean WER (asc)</option>
        <option value="mean_case_cer">Mean CER (asc)</option>
        <option value="num_samples">Samples (desc)</option>
        <option value="avg_time_per_sample_s">Speed (asc)</option>
      </select>
    </label>
  </div>

  <table id="leaderboard-table">
    <thead>
      <tr>
        <th>Rank</th>
        <th>Model</th>
        <th>Benchmark</th>
        <th onclick="sortBy('corpus_wer')" class="sortable">WER &#x25B4;&#x25BE;</th>
        <th onclick="sortBy('corpus_cer')" class="sortable">CER &#x25B4;&#x25BE;</th>
        <th onclick="sortBy('mean_case_wer')" class="sortable">Mean WER &#x25B4;&#x25BE;</th>
        <th onclick="sortBy('mean_case_cer')" class="sortable">Mean CER &#x25B4;&#x25BE;</th>
        <th onclick="sortBy('num_samples')" class="sortable">Samples</th>
        <th onclick="sortBy('avg_time_per_sample_s')" class="sortable">Sec/Sample</th>
      </tr>
    </thead>
    <tbody id="table-body">
    </tbody>
  </table>

  <div class="badge">
    <span class="badge-label">Leaderboard v{version}</span>
    <span class="badge-label">Best WER: <strong>{best_wer}</strong></span>
    <span class="badge-label">Models: <strong>{model_count}</strong></span>
  </div>

  <footer>
    <p>Built on the <a href="https://github.com/ericjohnemberda/phcodeswitch-ceb-dvo">PhCodeSwitch Cebu-Davao dataset</a>.</p>
    <p>To submit a model, run: <code>python run_leaderboard.py --model your_model_id</code></p>
  </footer>
</div>

<script>
{js}
</script>
</body>
</html>"""


CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, sans-serif;
  background: #0d1117; color: #c9d1d9; line-height: 1.6;
}
.container { max-width: 1200px; margin: 0 auto; padding: 24px; }
header { margin-bottom: 32px; }
header h1 {
  font-size: 2rem; font-weight: 700;
  background: linear-gradient(135deg, #58a6ff, #bc8cff);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
.subtitle { color: #8b949e; font-size: 1.1rem; margin-top: 4px; }
.timestamp { color: #484f58; font-size: 0.85rem; margin-top: 8px; }
.filters {
  display: flex; gap: 16px; align-items: center;
  margin-bottom: 20px; flex-wrap: wrap;
}
.filters label { color: #8b949e; font-size: 0.9rem; }
.filters select {
  background: #21262d; color: #c9d1d9; border: 1px solid #30363d;
  padding: 6px 12px; border-radius: 6px; font-size: 0.9rem; cursor: pointer;
}
.filters select:focus { outline: none; border-color: #58a6ff; }
table {
  width: 100%; border-collapse: collapse;
  background: #161b22; border: 1px solid #30363d; border-radius: 8px;
  overflow: hidden;
}
th {
  background: #21262d; padding: 12px 16px; text-align: left;
  font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em;
  color: #8b949e; border-bottom: 1px solid #30363d; white-space: nowrap;
}
th.sortable { cursor: pointer; user-select: none; }
th.sortable:hover { color: #58a6ff; }
td {
  padding: 12px 16px; border-bottom: 1px solid #21262d;
  font-size: 0.9rem;
}
tr:hover td { background: #1c2128; }
tr:last-child td { border-bottom: none; }
.rank {
  display: inline-block; width: 28px; height: 28px; line-height: 28px;
  text-align: center; border-radius: 50%; font-weight: 700; font-size: 0.8rem;
}
.rank-1 { background: #ffd700; color: #0d1117; }
.rank-2 { background: #c0c0c0; color: #0d1117; }
.rank-3 { background: #cd7f32; color: #0d1117; }
.rank-default { background: #21262d; color: #8b949e; }
.model-name { font-weight: 600; color: #58a6ff; }
.model-tag {
  display: inline-block; font-size: 0.7rem; padding: 2px 8px;
  border-radius: 12px; margin-left: 8px; font-weight: 500;
}
.tag-finetuned { background: #1b4332; color: #52b788; }
.tag-pretrained { background: #1b2838; color: #58a6ff; }
.wer { font-family: 'JetBrains Mono', monospace; font-weight: 600; }
.wer-good { color: #3fb950; }
.wer-mid { color: #d29922; }
.wer-bad { color: #f85149; }
.badge {
  display: flex; gap: 8px; margin-top: 24px; flex-wrap: wrap;
}
.badge-label {
  background: #21262d; padding: 6px 14px; border-radius: 20px;
  font-size: 0.8rem; color: #8b949e;
}
.badge-label strong { color: #c9d1d9; }
footer { margin-top: 48px; padding-top: 24px; border-top: 1px solid #21262d; color: #484f58; font-size: 0.85rem; }
footer a { color: #58a6ff; text-decoration: none; }
footer a:hover { text-decoration: underline; }
footer code { background: #21262d; padding: 2px 6px; border-radius: 4px; font-size: 0.8rem; }
"""


JS = """
const data = %DATA%;

function getMetric(row, key) {
  return row.metrics && row.metrics[key] !== undefined ? row.metrics[key] : null;
}

function getModelType(modelKey) {
  const ft = data.fine_tuned_models || [];
  return ft.includes(modelKey) ? 'fine-tuned' : 'pretrained';
}

function render() {
  const bm = document.getElementById('benchmark-select').value;
  const sortKey = document.getElementById('sort-select').value;

  let filtered = data.entries.filter(e => e.benchmark === bm);
  const desc = sortKey === 'num_samples';
  filtered.sort((a, b) => {
    const va = getMetric(a, sortKey) ?? 0;
    const vb = getMetric(b, sortKey) ?? 0;
    return desc ? vb - va : va - vb;
  });

  const tbody = document.getElementById('table-body');
  tbody.innerHTML = '';

  filtered.forEach((entry, idx) => {
    const rank = idx + 1;
    const m = entry.metrics;
    const werClass = m.corpus_wer < 15 ? 'wer-good' : m.corpus_wer < 30 ? 'wer-mid' : 'wer-bad';
    const modelType = getModelType(entry.model_key);
    const tagClass = modelType === 'fine-tuned' ? 'tag-finetuned' : 'tag-pretrained';

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><span class="rank rank-${rank <= 3 ? rank : 'default'}">${rank}</span></td>
      <td>
        <span class="model-name">${entry.model_name}</span>
        <span class="model-tag ${tagClass}">${modelType}</span>
      </td>
      <td>${entry.benchmark}</td>
      <td class="wer ${werClass}">${m.corpus_wer.toFixed(2)}%</td>
      <td class="wer">${m.corpus_cer.toFixed(2)}%</td>
      <td class="wer">${m.mean_case_wer.toFixed(2)}%</td>
      <td class="wer">${m.mean_case_cer.toFixed(2)}%</td>
      <td>${m.num_samples}</td>
      <td>${m.avg_time_per_sample_s.toFixed(2)}s</td>
    `;
    tbody.appendChild(tr);
  });
}

function filterTable() { render(); }
document.addEventListener('DOMContentLoaded', render);
"""


def generate_html(results_file: str, output_file: str):
    with open(results_file, encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("entries", [])
    title = data.get("name", "ASR Leaderboard")

    benchmarks = sorted(set(e["benchmark"] for e in entries))
    benchmark_options = "\n".join(
        f'<option value="{b}">{b}</option>' for b in benchmarks
    )

    fine_tuned_models = [
        e["model_key"] for e in entries
        if data.get("fine_tuned_keys", [])
    ]

    js_data = {
        "entries": entries,
        "fine_tuned_models": fine_tuned_models,
    }

    best_wer = "—"
    if entries:
        valid = [e for e in entries if e["metrics"]["corpus_wer"] is not None]
        if valid:
            best = min(valid, key=lambda e: e["metrics"]["corpus_wer"])
            best_wer = f"{best['metrics']['corpus_wer']:.2f}% ({best['model_name']}, {best['benchmark']})"

    html = HTML_TEMPLATE.format(
        title=title,
        css=CSS,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        entry_count=len(entries),
        benchmark_options=benchmark_options,
        version=data.get("version", "1.0.0"),
        best_wer=best_wer,
        model_count=len(set(e["model_key"] for e in entries)),
        js=JS.replace("%DATA%", json.dumps(js_data, indent=2)),
    )

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Leaderboard HTML generated: {output_path}")
    return str(output_path)
