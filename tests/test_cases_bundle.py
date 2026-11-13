"""Hard-case bundles: labels from the app state, case.json round trip, zip I/O."""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

from _case_state import fit_all, folder_state, set_manual
from bubbletrack.model.cases.build import (
    CaseRequest, crop_to_roi, default_file_name, estimate_bytes, make_case, plan_frames,
    write_case,
)
from bubbletrack.model.cases.bundle import (
    decode_frame, encode_frame, find_bundles, frame_file_name, open_bundle,
)
from bubbletrack.model.cases.labels import app_result, frame_label
from bubbletrack.model.cases.schema import (
    BY_ACCEPTED, BY_MANUAL, LABEL_BUBBLE, LABEL_NO_BUBBLE, LABEL_UNLABELLED,
    ROLE_CONTEXT, ROLE_REFERENCE, ROLE_SELECTED, AppResult, CaseFormatError, case_from_json,
    case_to_json,
)
from bubbletrack.model.image_io import imread_unicode
from bubbletrack.model.provenance import FitSource
from bubbletrack.model.state import update_state

ROI = ((31, 220), (61, 340))


# -- labels -------------------------------------------------------------------- #

class TestLabels:
    def test_manual_fit_is_a_bubble_label(self):
        label = frame_label(AppResult(30.0, (100.0, 120.0), source="manual"))
        assert (label.status, label.radius, label.center, label.by) == \
            (LABEL_BUBBLE, 30.0, (100.0, 120.0), BY_MANUAL)

    def test_locked_result_is_a_bubble_label(self):
        label = frame_label(AppResult(30.0, (100.0, 120.0), source="auto", locked=True))
        assert (label.status, label.by) == (LABEL_BUBBLE, BY_ACCEPTED)

    def test_locked_without_result_is_no_bubble(self):
        label = frame_label(AppResult(None, locked=True))
        assert (label.status, label.by) == (LABEL_NO_BUBBLE, BY_ACCEPTED)

    def test_a_circle_fix_is_a_bubble_label(self):
        """A drag-a-circle result is the user's decision, like a manual fit."""
        from bubbletrack.model.cases.schema import BY_GUIDED

        label = frame_label(AppResult(30.0, (100.0, 120.0), source="guided"))
        assert (label.status, label.by, label.radius) == (LABEL_BUBBLE, BY_GUIDED, 30.0)

    @pytest.mark.parametrize("result", [AppResult(25.0, (1.0, 1.0), source="auto"),
                                        AppResult(25.0, (1.0, 1.0), source="tracked"),
                                        AppResult(None)])
    def test_unverified_frames_are_unlabelled(self, result):
        assert not frame_label(result).is_labelled

    def test_app_result_reads_the_state(self, make_sequence):
        state = folder_state(make_sequence("A", 3))
        state.radius[0], state.circle_fit_par[0] = 31.5, [125.0, 200.0]
        state.fit_source[0], state.fit_locked[0] = FitSource.REFIT, True
        state.fit_confidence[0], state.fit_flags[0] = 0.4, frozenset({"jump", "collapse"})
        result = app_result(state, 0)
        assert result == AppResult(31.5, (125.0, 200.0), 0.4, ("collapse", "jump"), "refit", True)
        assert app_result(state, 1) == AppResult(None, None, None, (), "none", False)


# -- frame plan, crop, size ------------------------------------------------------ #

class TestPlan:
    def test_context_and_reference(self):
        plan = plan_frames(10, 12, 100, reference=50)
        assert [i for i, _ in plan] == [7, 8, 9, 10, 11, 12, 13, 14, 15, 50]
        assert dict(plan)[10] == ROLE_SELECTED and dict(plan)[7] == ROLE_CONTEXT
        assert dict(plan)[50] == ROLE_REFERENCE

    def test_clipped_to_the_recording_and_reversed_range(self):
        plan = plan_frames(4, 1, 6, reference=2)
        assert plan == ((0, ROLE_CONTEXT), (1, ROLE_SELECTED), (2, ROLE_SELECTED),
                        (3, ROLE_SELECTED), (4, ROLE_SELECTED), (5, ROLE_CONTEXT))

    def test_crop_and_size(self):
        raw = np.arange(20 * 30, dtype=np.uint16).reshape(20, 30)
        crop = crop_to_roi(raw, (2, 5), (3, 10))
        assert crop.shape == (4, 8) and crop[0, 0] == raw[1, 2]
        crop[0, 0] = 0
        assert raw[1, 2] != 0  # a copy
        assert estimate_bytes(10, (2, 5), (3, 10), raw) == 10 * 4 * 8 * 2
        rgb = np.zeros((20, 30, 3), np.uint8)
        assert estimate_bytes(1, (1, 20), (1, 30), rgb) == 20 * 30 * 3


# -- case.json --------------------------------------------------------------------- #

@pytest.fixture
def state(make_sequence):
    st = fit_all(folder_state(make_sequence("gel run", 12), roi=ROI))
    set_manual(st, 5, 40.0, 126.0, 199.0)
    st.fit_locked[6] = True
    st.radius[7], st.fit_source[7] = -1.0, FitSource.NONE
    st.fit_locked[7] = True  # accepted without a result: "no bubble here"
    st.radius[4], st.fit_source[4] = -1.0, FitSource.NONE  # e.g. deleted: not a label
    return st


def _request(**kw) -> CaseRequest:
    return CaseRequest(**{"start": 5, "end": 7, "note": "  merges with the wall ",
                          "reporter": " Ann ", "max_radius": 400.0, **kw})


class TestCaseJson:
    def test_make_case(self, state):
        case = make_case(state, _request(), (250, 400))
        assert case.selection == (5, 7) and case.offset == (30, 60)
        assert [f.index for f in case.frames] == list(range(2, 11))
        assert case.source["name"] == "gel run" and case.source["kind"] == "folder"
        assert "\\" not in json.dumps(dict(case.source)) and "/" not in case.source["name"]
        assert case.frame(5).label.by == BY_MANUAL and case.frame(6).label.by == BY_ACCEPTED
        assert (case.frame(7).label.status, case.frame(7).label.by) == (LABEL_NO_BUBBLE, BY_ACCEPTED)
        assert case.frame(8).label.status == LABEL_UNLABELLED
        assert case.frame(4).result.radius is None
        assert case.frame(4).label.status == LABEL_UNLABELLED  # no result, not accepted
        assert case.frame(5).source_file == "frame_006.tiff"
        assert (case.note, case.reporter) == ("merges with the wall", "Ann")
        assert case.detection.params["gridx"] == [31, 220]
        assert case.case_id.startswith("gel_run_f6-8_")

    def test_round_trip_is_strict_json(self, state):
        case = make_case(state, _request(), (250, 400))
        text = json.dumps(case_to_json(case), allow_nan=False)
        assert case_from_json(json.loads(text)) == case

    def test_infinite_max_radius_round_trips(self, state):
        case = make_case(state, _request(max_radius=float("inf")), (250, 400))
        data = json.loads(json.dumps(case_to_json(case), allow_nan=False))
        assert data["detection"]["max_radius"] is None
        assert case_from_json(data).detection.max_radius == float("inf")

    @pytest.mark.parametrize("edit, message", [
        (lambda d: d.update(format="other"), "not a BubbleTrack"),
        (lambda d: d.update(format_version=99), "unsupported"),
        (lambda d: d["frames"][0].update(role="other"), "role"),
        (lambda d: d["frames"][0].update(index=-1), "index"),
        (lambda d: d["frames"][1].update(index=d["frames"][0]["index"]), "twice"),
        (lambda d: d["frames"][3]["label"].update(radius=None), "radius"),
        (lambda d: d["detection"].update(mode="magic"), "mode"),
        (lambda d: d["roi"].update(gridx=[5, 2]), "gridx"),
        (lambda d: d.update(frames=[]), "frames"),
    ])
    def test_invalid_case_json_is_rejected(self, state, edit, message):
        data = case_to_json(make_case(state, _request(), (250, 400)))
        edit(data)
        with pytest.raises(CaseFormatError, match=message):
            case_from_json(data)


# -- bundle files --------------------------------------------------------------------- #

@pytest.mark.parametrize("image", [
    np.arange(12, dtype=np.uint16).reshape(3, 4) * 5000,
    np.arange(12, dtype=np.uint8).reshape(3, 4),
    np.linspace(0, 1, 12, dtype=np.float32).reshape(3, 4),
    np.arange(36, dtype=np.uint16).reshape(3, 4, 3) * 999,
])
def test_frames_are_stored_losslessly(image):
    ext, data = encode_frame(image)
    assert ext == ".tif"
    back = decode_frame("x" + ext, data)
    assert back.dtype == image.dtype and np.array_equal(back, image)


def test_unsupported_dtype_falls_back_to_npy(monkeypatch):
    import cv2

    monkeypatch.setattr(cv2, "imencode", lambda *a, **k: (False, None))
    image = np.arange(6, dtype=np.int64).reshape(2, 3)
    ext, data = encode_frame(image)
    assert ext == ".npy" and np.array_equal(decode_frame("f.npy", data), image)


def test_reference_frame_only_in_auto_mode(state):
    legacy = make_case(state, _request(reference_index=11), (250, 400))
    assert legacy.frame(11) is None and legacy.detection.reference_index is None
    auto = make_case(update_state(state, detection_mode="edge"),
                     _request(reference_index=11, tracking=True), (250, 400))
    assert auto.frame(11).role == ROLE_REFERENCE and auto.detection.reference_index == 11
    assert auto.detection.tracking and not legacy.detection.tracking
    out_of_range = make_case(update_state(state, detection_mode="edge"),
                             _request(reference_index=99), (250, 400))
    assert out_of_range.detection.reference_index is None


def test_write_and_open_bundle(state, tmp_path):
    state = update_state(state, detection_mode="edge")
    case = make_case(state, _request(reference_index=11), (250, 400))
    path = tmp_path / "out" / "gel case µ.zip"
    path.parent.mkdir()
    written, size = write_case(path, case, lambda i: imread_unicode(state.images[i]))
    assert size == path.stat().st_size > 0
    assert not list(path.parent.glob("*.part"))
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
    assert "case.json" in names and frame_file_name(11) in names and len(names) == 1 + 10
    with open_bundle(path) as bundle:
        assert bundle.case == written
        assert bundle.case.frame(11).role == ROLE_REFERENCE
        full = imread_unicode(state.images[5])
        assert np.array_equal(bundle.read_frame(5), full[30:220, 60:340])
        assert bundle.read_frame(5).dtype == np.uint16
        with pytest.raises(KeyError):
            bundle.read_frame(0)


def test_missing_image_writes_nothing(state, tmp_path):
    case = make_case(state, _request(), (250, 400))
    from bubbletrack.model.cases.bundle import write_bundle

    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(KeyError):
        write_bundle(out / "x.zip", case, {})
    assert list(out.iterdir()) == []


def _write(state, folder, name="c.zip"):
    path = folder / name
    write_case(path, make_case(state, _request(), (250, 400)), lambda i: imread_unicode(state.images[i]))
    return path


def test_unpacked_and_rezipped_bundles_read_the_same(state, tmp_path):
    path = _write(state, tmp_path)
    unpacked = tmp_path / "unpacked"
    with zipfile.ZipFile(path) as zf:
        zf.extractall(unpacked)
    rezipped = tmp_path / "rezipped.zip"
    with zipfile.ZipFile(rezipped, "w") as zf:
        for file in unpacked.rglob("*"):
            if file.is_file():
                zf.write(file, "my case/" + file.relative_to(unpacked).as_posix())
    with open_bundle(path) as a, open_bundle(unpacked) as b, open_bundle(rezipped) as c:
        assert a.case == b.case == c.case
        assert np.array_equal(a.read_frame(6), b.read_frame(6))
        assert np.array_equal(a.read_frame(6), c.read_frame(6))
    assert find_bundles(tmp_path) == sorted([path, rezipped, unpacked])


def test_folder_bundle_cannot_read_outside_itself(state, tmp_path):
    path = _write(state, tmp_path)
    unpacked = tmp_path / "unpacked"
    with zipfile.ZipFile(path) as zf:
        zf.extractall(unpacked)
    data = json.loads((unpacked / "case.json").read_text(encoding="utf-8"))
    data["frames"][0]["file"] = "../c.zip"
    (unpacked / "case.json").write_text(json.dumps(data), encoding="utf-8")
    with open_bundle(unpacked) as bundle, pytest.raises(CaseFormatError, match="outside"):
        bundle.read_frame(data["frames"][0]["index"])


@pytest.mark.parametrize("content", [b"not a zip", None])
def test_not_a_bundle(tmp_path, content):
    path = tmp_path / "x.zip"
    if content is None:
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("readme.txt", "hello")
    else:
        path.write_bytes(content)
    with pytest.raises(CaseFormatError):
        open_bundle(path)


def test_corrupted_member_is_a_format_error(state, tmp_path):
    path = _write(state, tmp_path)
    data = bytearray(path.read_bytes())
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo("frames/frame_000006.tif")
    start = info.header_offset + 30 + len(info.filename) + len(info.extra) + 20
    data[start:start + 40] = bytes(40)  # damage the deflated data
    path.write_bytes(bytes(data))
    with open_bundle(path) as bundle, pytest.raises(CaseFormatError, match="frame_000006"):
        bundle.read_frame(5)


def test_default_file_name(state):
    from datetime import datetime

    name = default_file_name(state, 4, 9, datetime(2026, 11, 12, 9, 30, 0))
    assert name == "gel_run_hardcase_f5-10_20261112_093000.zip"
