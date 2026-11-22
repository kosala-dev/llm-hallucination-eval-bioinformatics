"""Tiny FastAPI page that shows results_summary.json.  pip install fastapi uvicorn
Run:  uvicorn results_app:app --reload     then open http://127.0.0.1:8000"""
import json
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="LLM Eval Harness")


@app.get("/api/results")
def results():
    with open("results_summary.json") as f:
        return json.load(f)


@app.get("/", response_class=HTMLResponse)
def page():
    data = results()
    rows = "".join(
        f"<tr><td>{k}</td><td>{v.get('accuracy','')}</td><td>{v.get('halluc','')}</td>"
        f"<td>{v.get('abstain_rate','')}</td><td>{v.get('trap_score','')}</td></tr>"
        for k, v in data.items() if "|" in k)
    return (f"<h1>Llama 3.1 8B: bioinformatics eval</h1><table border=1 cellpadding=6>"
            f"<tr><th>split|config</th><th>accuracy</th><th>hallucinated</th><th>abstain rate</th>"
            f"<th>trap score</th></tr>{rows}</table>")
