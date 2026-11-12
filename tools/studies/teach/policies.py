"""Measurement policies evaluated offline on the study records.

A record holds, per test frame: plain Auto (``a0``), ML's region without and with
the half-contrast refinement (``reg_raw`` / ``reg_ref``, with confidence terms) and
Auto guided by each region (``g_raw`` / ``g_ref``).  A policy maps a record to
``(circle | None, accepted, branch)``; only accepted circles count.
"""

from __future__ import annotations

import numpy as np

import common  # noqa: F401  # puts the repo's src first on sys.path
from bubbletrack.model.teach.confidence import region_confidence

GATE = 0.3
AGREE_R, AGREE_C = (0.15, 3.0), (0.25, 4.0)      # package defaults (MeasureConfig)


def circ(d):
    return None if d is None else (d["R"], d["rc"], d["cc"])


def wall_agrees(reg: dict, edge: dict, agree_r=AGREE_R, agree_c=AGREE_C) -> bool:
    if reg is None or edge is None or not (np.isfinite(edge["R"]) and edge["R"] > 0):
        return False
    r = max(reg["R"], 1.0)
    return (abs(edge["R"] - reg["R"]) <= max(agree_r[0] * r, agree_r[1])
            and np.hypot(edge["rc"] - reg["rc"], edge["cc"] - reg["cc"]) <= max(agree_c[0] * r, agree_c[1]))


def conf_product(terms: dict) -> float:
    return region_confidence(terms)


def _ml(reg, conf_fn, gate, branch="ml"):
    if reg is None:
        return None, False, "none"
    return circ(reg), conf_fn(reg["terms"]) >= gate, branch


def make_policies(conf_fn=conf_product, gate=GATE, agree_r=AGREE_R, agree_c=AGREE_C) -> dict:
    """{name: fn(record, a0) -> (circle, accepted, branch)}."""

    def auto(rec, a0):
        return circ(a0), bool(a0["acc"]), "auto"

    def prior(region_key, guided_key):
        def fn(rec, a0):
            g = rec[guided_key]
            if g is None:
                return None, False, "none"
            return circ(g), bool(g["acc"]), "auto_ml"
        return fn

    def contour(region_key):
        def fn(rec, a0):
            return _ml(rec[region_key], conf_fn, gate)
        return fn

    def combined(region_key, guided_key, check=True):
        """Owner's precedence: Auto inside the ML region, else ML's outline, else nothing."""
        def fn(rec, a0):
            reg, g = rec[region_key], rec[guided_key]
            if reg is None:
                return None, False, "none"
            if g is not None and g["acc"] and (not check or wall_agrees(reg, g, agree_r, agree_c)):
                return circ(g), True, "auto_ml"
            return _ml(reg, conf_fn, gate)
        return fn

    def guarded(region_key, guided_key):
        """Owner's precedence, but an accepted plain-Auto result is never replaced by an
        ML outline: Auto (ML-guided) -> plain Auto (if accepted) -> ML outline -> nothing."""
        def fn(rec, a0):
            reg, g = rec[region_key], rec[guided_key]
            if reg is not None and g is not None and g["acc"] and wall_agrees(reg, g, agree_r, agree_c):
                return circ(g), True, "auto_ml"
            if a0["acc"]:
                return circ(a0), True, "auto"
            return _ml(reg, conf_fn, gate)
        return fn

    def auto_first(region_key, guided_key):
        """Variant: plain Auto when it is accepted, else the combined precedence."""
        inner = combined(region_key, guided_key)

        def fn(rec, a0):
            if a0["acc"]:
                return circ(a0), True, "auto"
            return inner(rec, a0)
        return fn

    return {
        "auto": auto,
        "i_prior": prior("reg_ref", "g_ref"),
        "i_prior_raw": prior("reg_raw", "g_raw"),
        "ii_contour": contour("reg_raw"),
        "ii_refined": contour("reg_ref"),
        "ii_ref_k1": contour("reg_k1"),
        "ii_ref_k4": contour("reg_k4"),
        "ii_ref_b10": contour("reg_b10"),
        "ii_ref_b25": contour("reg_b25"),
        "ii_ref_2p": contour("reg_2p"),
        "iii_combined": combined("reg_ref", "g_ref"),
        "iii_combined_raw": combined("reg_raw", "g_raw"),
        "iii_nocheck": combined("reg_ref", "g_ref", check=False),
        "iv_auto_first": auto_first("reg_ref", "g_ref"),
        "v_guarded": guarded("reg_ref", "g_ref"),
    }
