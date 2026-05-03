"""
Minimal Flask annotation app for human evaluation of summaries.
Annotators score each summary on 4 dimensions (1–5 Likert).

Usage:
  ANNOTATOR_ID=ann1 python src/annotate_app.py
  # Open http://localhost:5000 in browser

Data is saved incrementally to:
  data/annotations/raw/{annotator_id}_annotations.json
"""

import json
import os
import random
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template_string, request, url_for

DATA_DIR = Path(__file__).parent.parent / "data"
ANN_DIR = DATA_DIR / "annotations" / "raw"
ANN_DIR.mkdir(parents=True, exist_ok=True)

ANNOTATOR_ID = os.environ.get("ANNOTATOR_ID", "ann1")
SEED = hash(ANNOTATOR_ID) % 1000  # different random order per annotator

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Summary Annotation — {{ annotator }}</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; }
    .progress { color: #888; margin-bottom: 1em; }
    .pane { display: flex; gap: 2em; }
    .article { flex: 1; background: #f8f8f8; padding: 1em; border-radius: 6px; max-height: 400px; overflow-y: auto; font-size: 0.9em; }
    .summary-block { flex: 1; }
    .summary-text { background: #e8f4fd; padding: 1em; border-radius: 6px; margin-bottom: 1em; }
    .dim { margin: 0.7em 0; }
    .dim label { display: block; font-weight: bold; margin-bottom: 0.3em; }
    .dim .desc { font-size: 0.85em; color: #555; margin-bottom: 0.3em; }
    .scale { display: flex; gap: 0.5em; }
    .scale input[type=radio] { display: none; }
    .scale label.radio { padding: 6px 14px; border: 1px solid #ccc; border-radius: 4px; cursor: pointer; }
    .scale input[type=radio]:checked + label.radio { background: #2196F3; color: white; border-color: #2196F3; }
    .submit-btn { margin-top: 2em; background: #4CAF50; color: white; padding: 10px 30px;
                  border: none; border-radius: 4px; cursor: pointer; font-size: 1em; }
    .done { text-align: center; padding: 3em; }
  </style>
</head>
<body>
  <h2>Summary Annotation — Annotator: <b>{{ annotator }}</b></h2>
  <p class="progress">Example {{ current }} of {{ total }} | Summary {{ summary_idx }} ({{ model_label }})</p>
  <form method="POST" action="/submit">
    <input type="hidden" name="pair_id" value="{{ pair_id }}">
    <input type="hidden" name="summary_key" value="{{ summary_key }}">
    <div class="pane">
      <div>
        <h3>Article</h3>
        <div class="article">{{ article }}</div>
      </div>
      <div class="summary-block">
        <h3>Summary</h3>
        <div class="summary-text">{{ summary }}</div>
        {% for dim, desc in dimensions %}
        <div class="dim">
          <label>{{ dim.capitalize() }}</label>
          <div class="desc">{{ desc }}</div>
          <div class="scale">
            {% for v in [1,2,3,4,5] %}
            <input type="radio" id="{{ dim }}_{{ v }}" name="{{ dim }}" value="{{ v }}" required>
            <label class="radio" for="{{ dim }}_{{ v }}">{{ v }}</label>
            {% endfor %}
          </div>
        </div>
        {% endfor %}
      </div>
    </div>
    <button class="submit-btn" type="submit">Save &amp; Next →</button>
  </form>
</body>
</html>
"""

DONE_HTML = """
<!DOCTYPE html><html><body>
<div class="done" style="text-align:center;padding:3em;font-family:Arial">
<h2>✓ All done! Thank you, {{ annotator }}.</h2>
<p>Your annotations have been saved to the project.</p>
</div></body></html>
"""

DIMENSION_DESCS = [
    ("coherence", "Is the summary well-structured and logically organized? (1=incoherent, 5=perfectly coherent)"),
    ("consistency", "Are all facts consistent with the article? No hallucinations? (1=many errors, 5=fully accurate)"),
    ("fluency", "Is the language grammatically correct and natural? (1=unreadable, 5=perfect)"),
    ("relevance", "Does it cover the most important points? No trivial details? (1=irrelevant, 5=covers all key points)"),
]


def load_queue():
    """Build ordered list of (pair, summary_key) to annotate."""
    pairs_path = DATA_DIR / "system_outputs" / "all_pairs.json"
    with open(pairs_path) as f:
        pairs = json.load(f)

    queue = []
    for pair in pairs:
        queue.append((pair, "summary_A"))
        queue.append((pair, "summary_B"))

    # Also add adversarial
    adv_path = DATA_DIR / "adversarial" / "adversarial_examples.json"
    if adv_path.exists():
        with open(adv_path) as f:
            adversarial = json.load(f)
        for adv in adversarial:
            # Convert adversarial to pair-like format
            fake_pair_good = {
                "id": f"adv_{adv['id']}_good",
                "article": adv["article"],
                "summary_A": adv["good_summary"],
                "model_A": "good",
            }
            fake_pair_adv = {
                "id": f"adv_{adv['id']}_bad",
                "article": adv["article"],
                "summary_A": adv["adversarial_summary"],
                "model_A": f"adv_{adv['adversarial_type']}",
            }
            queue.append((fake_pair_good, "summary_A"))
            queue.append((fake_pair_adv, "summary_A"))

    random.seed(SEED)
    random.shuffle(queue)
    return queue


def load_done():
    out_path = ANN_DIR / f"{ANNOTATOR_ID}_annotations.json"
    if out_path.exists():
        with open(out_path) as f:
            records = json.load(f)
        return {(r["pair_id"], r["summary_key"]) for r in records}
    return set()


def save_annotation(record: dict):
    out_path = ANN_DIR / f"{ANNOTATOR_ID}_annotations.json"
    records = []
    if out_path.exists():
        with open(out_path) as f:
            records = json.load(f)
    records.append(record)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)


QUEUE = None


def get_queue():
    global QUEUE
    if QUEUE is None:
        QUEUE = load_queue()
    return QUEUE


@app.route("/")
def index():
    queue = get_queue()
    done = load_done()
    remaining = [(pair, skey) for pair, skey in queue if (pair["id"], skey) not in done]
    if not remaining:
        return render_template_string(DONE_HTML, annotator=ANNOTATOR_ID)

    pair, summary_key = remaining[0]
    current = len(done) + 1
    total = len(queue)
    model_label = pair.get(f"model_{summary_key[-1]}", "")

    return render_template_string(
        HTML,
        annotator=ANNOTATOR_ID,
        current=current,
        total=total,
        pair_id=pair["id"],
        summary_key=summary_key,
        article=pair["article"][:2500],
        summary=pair[summary_key],
        model_label=model_label,
        summary_idx="A" if summary_key == "summary_A" else "B",
        dimensions=DIMENSION_DESCS,
    )


@app.route("/submit", methods=["POST"])
def submit():
    record = {
        "id": f"{request.form['pair_id']}_{request.form['summary_key']}",
        "pair_id": request.form["pair_id"],
        "summary_key": request.form["summary_key"],
        "annotator": ANNOTATOR_ID,
        "coherence": int(request.form["coherence"]),
        "consistency": int(request.form["consistency"]),
        "fluency": int(request.form["fluency"]),
        "relevance": int(request.form["relevance"]),
    }
    save_annotation(record)
    return redirect(url_for("index"))


if __name__ == "__main__":
    print(f"Starting annotation app for {ANNOTATOR_ID}")
    print(f"Open http://localhost:5000 in your browser")
    app.run(debug=False, port=5000)
