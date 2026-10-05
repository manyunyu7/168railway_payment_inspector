"""
Labeling UI — browse downloaded payment proofs and assign clean labels.

Usage:
    python scripts/04_labeling_server.py
    → open http://localhost:5055

Keyboard shortcuts (in browser):
    A = valid (real receipt)
    S = fake  (not a valid payment proof)
    D = ambiguous / skip
    Z = undo last
    ← / → = prev / next (without labeling)

State is saved to data/manifest/human_labels.csv after every action.
Resumable: re-running the server picks up where you left off.
"""
import os
import csv
import json
import random
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest" / "manifest.csv"
LABELS_CSV = ROOT / "data" / "manifest" / "human_labels.csv"

PORT = 5055
LOCK = threading.Lock()


def load_queue() -> list[dict]:
    """Build the review queue: all downloaded images, enriched with manifest info."""
    # map filename -> manifest row
    mani: dict[str, dict] = {}
    with MANIFEST.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fn = r["payment_proof"].rsplit("/", 1)[-1]
            mani[fn] = r

    buckets = {"completed": [], "rejected": []}
    for status in ("completed", "rejected"):
        d = RAW / status
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if not p.is_file():
                continue
            m = mani.get(p.name, {})
            buckets[status].append({
                "filename": p.name,
                "status": status,
                "rel_path": f"/img/{status}/{p.name}",
                "auto_label": m.get("label", "?"),
                "reason": m.get("rejection_reason", ""),
                "amount": m.get("amount", ""),
                "pm_id": m.get("payment_method_id", ""),
                "created_at": m.get("created_at", ""),
            })
    # deterministic shuffle per bucket, then concatenate: completed FIRST, then rejected
    random.Random(42).shuffle(buckets["completed"])
    random.Random(42).shuffle(buckets["rejected"])
    return buckets["completed"] + buckets["rejected"]


def load_labels() -> dict[str, str]:
    out: dict[str, str] = {}
    if LABELS_CSV.exists():
        with LABELS_CSV.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out[r["filename"]] = r["human_label"]
    return out


def save_label(filename: str, status: str, auto_label: str, human_label: str):
    LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    existing = load_labels()
    existing[filename] = human_label
    all_rows = {}
    if LABELS_CSV.exists():
        with LABELS_CSV.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                all_rows[r["filename"]] = r
    all_rows[filename] = {
        "filename": filename, "status": status,
        "auto_label": auto_label, "human_label": human_label,
    }
    with LABELS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "status", "auto_label", "human_label"])
        w.writeheader()
        w.writerows(all_rows.values())


QUEUE: list[dict] = []
LABELS: dict[str, str] = {}


HTML = r"""<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Payment Proof Labeling</title>
<style>
  :root {
    --bg: #0f1419; --panel: #1a2029; --panel2: #232b36;
    --border: #2d3744; --text: #e6edf3; --muted: #8b949e;
    --accent: #58a6ff; --valid: #3fb950; --fake: #f85149; --ambig: #d29922;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
  header { background: var(--panel); border-bottom: 1px solid var(--border); padding: 10px 20px;
           display: flex; align-items: center; gap: 20px; }
  header h1 { font-size: 15px; font-weight: 600; color: var(--muted); letter-spacing: 0.5px; }
  .stats { display: flex; gap: 12px; font-size: 13px; margin-left: auto; }
  .stat { padding: 4px 10px; border-radius: 6px; background: var(--panel2); }
  .stat .n { font-weight: 700; }
  .stat.valid .n { color: var(--valid); }
  .stat.fake .n { color: var(--fake); }
  .stat.ambig .n { color: var(--ambig); }
  .progress-bar { height: 4px; background: var(--panel); position: relative; }
  .progress-bar .fill { height: 100%; background: linear-gradient(90deg, var(--valid), var(--accent)); transition: width 0.3s; }

  main { flex: 1; display: grid; grid-template-columns: 1fr 320px; overflow: hidden; }

  .viewer { background: #000; display: flex; align-items: center; justify-content: center;
            position: relative; overflow: hidden; padding: 20px; }
  .viewer img { max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 8px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.5); }
  .viewer .counter { position: absolute; top: 20px; left: 20px; background: rgba(0,0,0,0.6);
                     padding: 6px 12px; border-radius: 20px; font-size: 13px; font-family: ui-monospace, monospace; }

  .sidebar { background: var(--panel); border-left: 1px solid var(--border);
             display: flex; flex-direction: column; overflow-y: auto; }
  .meta { padding: 20px; border-bottom: 1px solid var(--border); }
  .meta h2 { font-size: 11px; text-transform: uppercase; color: var(--muted); letter-spacing: 1px;
             margin-bottom: 12px; }
  .meta-row { margin-bottom: 10px; }
  .meta-row .k { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
  .meta-row .v { font-size: 14px; margin-top: 2px; word-break: break-word; font-family: ui-monospace, monospace; }
  .meta-row .v.reason { font-family: inherit; font-style: italic; color: #d29922; }

  .badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 11px;
           font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
  .badge.valid { background: rgba(63,185,80,0.15); color: var(--valid); }
  .badge.fake { background: rgba(248,81,73,0.15); color: var(--fake); }
  .badge.ambig { background: rgba(210,153,34,0.15); color: var(--ambig); }
  .badge.auto { background: rgba(88,166,255,0.1); color: var(--accent); }

  .actions { padding: 20px; }
  .btn { width: 100%; padding: 14px; border: 2px solid; background: transparent;
         color: var(--text); border-radius: 10px; font-size: 15px; font-weight: 600;
         cursor: pointer; margin-bottom: 10px; display: flex; align-items: center;
         justify-content: space-between; transition: all 0.1s; }
  .btn:hover { transform: translateY(-1px); }
  .btn:active { transform: translateY(0); }
  .btn .kbd { background: rgba(255,255,255,0.08); padding: 2px 8px; border-radius: 4px;
              font-size: 12px; font-family: ui-monospace, monospace; }
  .btn-valid { border-color: var(--valid); }
  .btn-valid:hover { background: rgba(63,185,80,0.1); }
  .btn-fake { border-color: var(--fake); }
  .btn-fake:hover { background: rgba(248,81,73,0.1); }
  .btn-ambig { border-color: var(--ambig); }
  .btn-ambig:hover { background: rgba(210,153,34,0.1); }
  .btn-skip { border-color: var(--border); color: var(--muted); }

  .nav { display: flex; gap: 8px; padding: 0 20px 20px; }
  .nav .btn { margin: 0; padding: 10px; font-size: 13px; }

  .current-label { padding: 15px 20px; border-bottom: 1px solid var(--border);
                   display: flex; align-items: center; gap: 10px; font-size: 13px; }
  .current-label .k { color: var(--muted); }

  .empty { display: flex; align-items: center; justify-content: center; height: 100%;
           font-size: 18px; color: var(--muted); text-align: center; padding: 40px; }
</style>
</head>
<body>
<header>
  <h1>★ Payment Proof Labeling</h1>
  <div class="stats">
    <div class="stat"><span class="n" id="s-done">0</span> / <span id="s-total">0</span></div>
    <div class="stat valid">✓ <span class="n" id="s-valid">0</span></div>
    <div class="stat fake">✗ <span class="n" id="s-fake">0</span></div>
    <div class="stat ambig">? <span class="n" id="s-ambig">0</span></div>
  </div>
</header>
<div class="progress-bar"><div class="fill" id="progress" style="width: 0%"></div></div>

<main id="main">
  <div class="viewer">
    <div class="counter" id="counter">—</div>
    <img id="img" src="" alt="">
  </div>
  <aside class="sidebar">
    <div class="current-label">
      <span class="k">Human label:</span>
      <span id="current-label" class="badge ambig">belum</span>
    </div>
    <div class="meta">
      <h2>Info</h2>
      <div class="meta-row"><div class="k">Status DB</div><div class="v" id="m-status">—</div></div>
      <div class="meta-row"><div class="k">Auto label</div><div class="v"><span id="m-auto" class="badge auto">—</span></div></div>
      <div class="meta-row" id="row-reason"><div class="k">Rejection reason</div><div class="v reason" id="m-reason">—</div></div>
      <div class="meta-row"><div class="k">Amount</div><div class="v" id="m-amount">—</div></div>
      <div class="meta-row"><div class="k">Payment method ID</div><div class="v" id="m-pm">—</div></div>
      <div class="meta-row"><div class="k">Created</div><div class="v" id="m-created">—</div></div>
      <div class="meta-row"><div class="k">Filename</div><div class="v" id="m-filename" style="font-size:11px">—</div></div>
    </div>
    <div class="actions">
      <button class="btn btn-valid" onclick="label('valid')">
        ✓ Valid (receipt asli) <span class="kbd">A</span>
      </button>
      <button class="btn btn-fake" onclick="label('fake')">
        ✗ Fake (bukan bukti valid) <span class="kbd">S</span>
      </button>
      <button class="btn btn-ambig" onclick="label('ambiguous')">
        ? Ambigu / skip <span class="kbd">D</span>
      </button>
    </div>
    <div class="nav">
      <button class="btn btn-skip" onclick="nav(-1)">← Prev</button>
      <button class="btn btn-skip" onclick="nav(1)">Next →</button>
    </div>
  </aside>
</main>

<script>
let state = { idx: 0, items: [], labels: {}, stats: {} };

async function boot() {
  const r = await fetch('/api/state');
  state = await r.json();
  // jump to first unlabeled
  for (let i = 0; i < state.items.length; i++) {
    if (!state.labels[state.items[i].filename]) { state.idx = i; break; }
  }
  render();
}

function render() {
  if (state.items.length === 0) {
    document.getElementById('main').innerHTML = '<div class="empty">Belum ada image di-download.<br>Jalankan <code>scripts/02_download_images.py</code> dulu.</div>';
    return;
  }
  const it = state.items[state.idx];
  document.getElementById('counter').textContent = `${state.idx + 1} / ${state.items.length}`;
  document.getElementById('img').src = it.rel_path;
  document.getElementById('m-status').textContent = it.status;
  const auto = document.getElementById('m-auto');
  auto.textContent = it.auto_label;
  auto.className = 'badge ' + (it.auto_label.startsWith('valid') ? 'valid' : 'fake');
  document.getElementById('m-reason').textContent = it.reason || '—';
  document.getElementById('row-reason').style.display = it.reason ? 'block' : 'none';
  document.getElementById('m-amount').textContent = it.amount ? 'Rp ' + Number(it.amount).toLocaleString('id-ID') : '—';
  document.getElementById('m-pm').textContent = it.pm_id || '—';
  document.getElementById('m-created').textContent = it.created_at || '—';
  document.getElementById('m-filename').textContent = it.filename;

  const cur = state.labels[it.filename];
  const curEl = document.getElementById('current-label');
  if (cur) {
    curEl.textContent = cur;
    curEl.className = 'badge ' + (cur === 'valid' ? 'valid' : cur === 'fake' ? 'fake' : 'ambig');
  } else {
    curEl.textContent = 'belum';
    curEl.className = 'badge ambig';
  }
  updateStats();
}

function updateStats() {
  const vals = Object.values(state.labels);
  const v = vals.filter(x => x === 'valid').length;
  const f = vals.filter(x => x === 'fake').length;
  const a = vals.filter(x => x === 'ambiguous').length;
  const done = vals.length;
  document.getElementById('s-done').textContent = done;
  document.getElementById('s-total').textContent = state.items.length;
  document.getElementById('s-valid').textContent = v;
  document.getElementById('s-fake').textContent = f;
  document.getElementById('s-ambig').textContent = a;
  document.getElementById('progress').style.width = (done / state.items.length * 100) + '%';
}

async function label(l) {
  const it = state.items[state.idx];
  state.labels[it.filename] = l;
  await fetch('/api/label', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({filename: it.filename, status: it.status, auto_label: it.auto_label, human_label: l}),
  });
  nav(1);
}

function nav(delta) {
  const n = state.idx + delta;
  if (n < 0 || n >= state.items.length) return;
  state.idx = n;
  render();
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'a' || e.key === 'A') label('valid');
  else if (e.key === 's' || e.key === 'S') label('fake');
  else if (e.key === 'd' || e.key === 'D') label('ambiguous');
  else if (e.key === 'ArrowRight') nav(1);
  else if (e.key === 'ArrowLeft') nav(-1);
});

boot();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a, **k): pass

    def _json(self, data, code=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/state":
            return self._json({"items": QUEUE, "labels": LABELS, "idx": 0, "stats": {}})

        if self.path.startswith("/img/"):
            # /img/<status>/<filename>
            parts = self.path[len("/img/"):].split("/", 1)
            if len(parts) != 2:
                self.send_response(400); self.end_headers(); return
            status, fn = parts
            if status not in ("completed", "rejected"):
                self.send_response(400); self.end_headers(); return
            p = RAW / status / fn
            if not p.exists():
                self.send_response(404); self.end_headers(); return
            # content type by extension
            ct = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                  "webp": "image/webp", "jfif": "image/jpeg", "heic": "image/heic"}.get(
                p.suffix.lstrip(".").lower(), "application/octet-stream")
            data = p.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_response(404); self.end_headers()

    def do_POST(self):
        if self.path != "/api/label":
            self.send_response(404); self.end_headers(); return
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n).decode("utf-8"))
        with LOCK:
            LABELS[body["filename"]] = body["human_label"]
            save_label(body["filename"], body["status"], body["auto_label"], body["human_label"])
        self._json({"ok": True})


def main():
    global QUEUE, LABELS
    QUEUE = load_queue()
    LABELS = load_labels()
    print(f"Loaded {len(QUEUE)} images into queue, {len(LABELS)} already labeled")
    print(f"→ Open http://localhost:{PORT}")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
