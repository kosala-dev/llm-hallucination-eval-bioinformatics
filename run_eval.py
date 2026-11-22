"""Run the evaluation on a local Llama 3.1 8B through Ollama.

  python run_eval.py --split dev                       # tune prompt B here, as often as you like (use --force)
  python run_eval.py --split test                      # run ONCE, after prompts are frozen
  python run_eval.py --split test --temp 0.7 --runs 3 --tag stability   # optional stability check

Each run writes scores_<split>[_<tag>].csv with empty columns you fill by hand:
  score (0/1/2), abstain (0/1), halluc (0/1), error_category (for wrong or hallucinated rows)
"""
import argparse, datetime, importlib.metadata as md, json, os, re
import pandas as pd
import ollama

MODEL = "llama3.1:8b"
PROMPTS = {
    "A": None,  # baseline: no system prompt
    "B": ("You are a careful bioinformatics assistant. If you are not sure, or you cannot "
          "verify that what is asked about exists, say \"I don't know\". Never invent details."),
}
ABSTAIN_RE = re.compile(r"i don'?t know|i'?m not sure|cannot verify|can'?t verify|not aware of|no (such|record)", re.I)


def ask(q, system, temp, seed):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}]
    r = ollama.chat(model=MODEL, messages=msgs, options={"temperature": temp, "seed": seed})
    return r["message"]["content"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output file")
    a = ap.parse_args()

    out = f"scores_{a.split}{'_' + a.tag if a.tag else ''}.csv"
    if os.path.exists(out) and not a.force:
        raise SystemExit(f"{out} exists. Test split should be run once per configuration. Use --force only if you must.")

    qs = pd.read_csv("questions.csv")
    qs = qs[qs.split == a.split].copy()
    rows = []
    for cfg, system in PROMPTS.items():
        for run in range(a.runs):
            for _, q in qs.iterrows():
                ans = ask(q.question, system, a.temp, seed=run)
                rows.append({**q.to_dict(), "config": cfg, "run": run, "answer": ans,
                             "auto_abstain_guess": int(bool(ABSTAIN_RE.search(ans))),
                             "score": "", "abstain": "", "halluc": "", "error_category": ""})
                print(cfg, run, q.id, "->", ans[:70].replace("\n", " "))
    pd.DataFrame(rows).to_csv(out, index=False)

    meta = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "model": MODEL,
            "ollama_python": md.version("ollama"), "pandas": md.version("pandas"),
            "temperature": a.temp, "runs": a.runs, "split": a.split, "prompts": PROMPTS}
    try:
        meta["model_info"] = str(ollama.show(MODEL).get("details", ""))
    except Exception as e:  # keep going if the server doesn't return details
        meta["model_info"] = f"unavailable: {e}"
    with open("run_meta.jsonl", "a") as f:
        f.write(json.dumps(meta) + "\n")
    print(f"\nWrote {out}. Now score it by hand (hide the config column or shuffle rows to stay blind).")


if __name__ == "__main__":
    main()
