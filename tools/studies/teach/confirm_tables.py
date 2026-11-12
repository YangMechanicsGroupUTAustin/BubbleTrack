"""Tables of the ``confirm`` phase: the package's ``measure_frame`` with its final defaults.

Correct = accepted and within the label criterion; "w" = accepted but wrong.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import common as C

SOURCES = ("auto_ml", "auto", "ml")
LABELS = {"auto_ml": "Auto (ML-guided)", "auto": "plain Auto", "ml": "ML outline"}


def load_confirm(out: Path, recs: dict) -> dict:
    by_file = {k.replace("/", "_"): k for k in recs}
    data = {}
    for path in sorted((out / "confirm").glob("*.jsonl")):
        stem, cid = path.stem.split("__")
        lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        data[(by_file[stem], cid)] = [x for x in lines if x["type"] == "frame"]
    return data


def _ok(res: dict, label, radius_only: bool) -> bool:
    return bool(res["acc"]) and C.agree((res["R"], res["rc"], res["cc"]), label, radius_only)


def score_confirm(frames: list[dict], rec, auto: dict) -> dict:
    """{"auto"|"combined"|"guarded": dict(ok, wrong, n)} plus guarded branch counts."""
    out = {k: dict(ok=0, wrong=0, n=len(frames)) for k in ("auto", "combined", "guarded")}
    branches = Counter()
    for x in frames:
        label = rec.labels[x["frame"]]
        a0 = auto[str(x["frame"])]
        results = {"auto": a0, "combined": x["combined"], "guarded": x["guarded"]}
        for name, res in results.items():
            good = _ok(res, label, rec.radius_only)
            out[name]["ok"] += good
            out[name]["wrong"] += bool(res["acc"]) and not good
        g = x["guarded"]
        state = "ok" if _ok(g, label, rec.radius_only) else ("wrong" if g["acc"] else "rejected")
        branches[(g["source"], state)] += 1
    out["branches"] = branches
    return out


def _cell(s: dict) -> str:
    if not s["n"]:
        return "-"
    return f"{s['ok']}/{s['n']}" + (f" ({s['wrong']}w)" if s["wrong"] else "")


def _table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def per_recording(data: dict, recs: dict, autos: dict, cid: str) -> str:
    rows = []
    tot = {g: defaultdict(lambda: dict(ok=0, wrong=0, n=0)) for g in ("good", "hard")}
    for key, rec in recs.items():
        frames = data.get((key, cid))
        if not frames:
            continue
        s = score_confirm(frames, rec, autos[key])
        b = s["branches"]
        trip = ["/".join(str(b[(src, k)]) for k in ("ok", "wrong", "rejected")) for src in SOURCES]
        rows.append([key, _cell(s["auto"]), _cell(s["combined"]), _cell(s["guarded"])] + trip
                    + [b[("none", "rejected")]])
        group = "good" if key in C.GOOD_FOR_AUTO else "hard"
        for name in ("auto", "combined", "guarded"):
            for k in ("ok", "wrong", "n"):
                tot[group][name][k] += s[name][k]
    for g in ("good", "hard"):
        rows.append([f"**{g} total**"] + [f"**{_cell(tot[g][n])}**" for n in ("auto", "combined", "guarded")]
                    + ["", "", "", ""])
    heads = ["recording", "Auto", "combined (owner)", "guarded (default)"] + [
        f"{LABELS[s]} ok/w/rej" for s in SOURCES] + ["no result"]
    return _table(heads, rows)


def overview(data: dict, recs: dict, autos: dict, cids) -> str:
    rows = []
    for cid in cids:
        tot = {g: defaultdict(lambda: dict(ok=0, wrong=0, n=0)) for g in ("good", "hard")}
        for key, rec in recs.items():
            frames = data.get((key, cid))
            if not frames:
                continue
            s = score_confirm(frames, rec, autos[key])
            group = "good" if key in C.GOOD_FOR_AUTO else "hard"
            for name in ("auto", "combined", "guarded"):
                for k in ("ok", "wrong", "n"):
                    tot[group][name][k] += s[name][k]
        rows.append([cid] + [_cell(tot[g][n]) for g in ("good", "hard") for n in ("auto", "combined", "guarded")])
    return _table(["taught frames", "good: Auto", "good: combined", "good: guarded",
                   "hard: Auto", "hard: combined", "hard: guarded"], rows)


def timing_from_confirm(data: dict) -> str:
    ms = sorted(x["ms"] for frames in data.values() for x in frames)
    if not ms:
        return "(no confirm data)"
    median, p90 = ms[len(ms) // 2], ms[min(len(ms) - 1, int(0.9 * len(ms)))]
    return (f"measure_frame (single-threaded, 6 study processes in parallel): median {median:.0f} ms, "
            f"90th percentile {p90:.0f} ms over {len(ms)} frames")
