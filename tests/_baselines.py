"""Per-frame results of the local real-data tests, tracked in ``tests/baselines/``.

The totals of those tests only say how many frames are right; these files say
which.  Any frame that moves fails its test until the file is regenerated
(``BUBBLETRACK_UPDATE_BASELINES=1``) and committed with the change, so the
diff shows exactly which frames an engine or tracking change moved.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from bubbletrack.model.edge import ENGINE_VERSION

BASELINE_DIR = Path(__file__).with_name("baselines")
UPDATE_ENV = "BUBBLETRACK_UPDATE_BASELINES"


def check(name: str, found: dict[str, dict[str, list[int]]]) -> None:
    """*found* (per sequence: named lists of frame numbers) must equal ``baselines/<name>.json``.

    Fails on any change, gains included; with ``BUBBLETRACK_UPDATE_BASELINES=1``
    (or no file yet) the file is (re)written instead.
    """
    path = BASELINE_DIR / f"{name}.json"
    found = {seq: {key: sorted(int(f) for f in frames) for key, frames in lists.items()}
             for seq, lists in found.items()}
    data = {"engine": ENGINE_VERSION, "sequences": {k: found[k] for k in sorted(found)}}
    if os.environ.get(UPDATE_ENV) == "1" or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        return
    saved = json.loads(path.read_text(encoding="utf-8"))
    changes = []
    for seq in sorted(set(saved["sequences"]) | set(found)):
        old, new = saved["sequences"].get(seq, {}), found.get(seq, {})
        for key in sorted(set(old) | set(new)):
            lost = sorted(set(old.get(key, [])) - set(new.get(key, [])))
            won = sorted(set(new.get(key, [])) - set(old.get(key, [])))
            if lost or won:
                changes.append(f"{seq} {key}: -{lost} +{won}")
    assert not changes, (f"results differ from {path.name} ({saved.get('engine')} -> {ENGINE_VERSION}); "
                         f"if intended, rerun with {UPDATE_ENV}=1 and commit it:\n" + "\n".join(changes))
