"""Replaying a hard case: the current engine on the stored ROI crops."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from _case_state import fit_all, folder_state, max_radius
from bubbletrack.model import edge_pipeline
from bubbletrack.model.cases.build import CaseRequest, make_case, write_case
from bubbletrack.model.cases.bundle import open_bundle
from bubbletrack.model.cases.replay import replay_case
from bubbletrack.model.cases.schema import CaseFormatError
from bubbletrack.model.image_io import imread_unicode
from conftest import write_sequence

ROI = ((31, 220), (61, 340))
RADII = (0.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0)  # frame 1 has no bubble: the reference
LEGACY = {"img_thr": 0.58, "removing_factor": 85, "gaussian_sigma": 3.0}


def _bundle(state, tmp_path, *, start=3, end=4, reference=None, tracking=False, name="c.zip"):
    case = make_case(state, CaseRequest(start=start, end=end, reference_index=reference,
                                        tracking=tracking, max_radius=max_radius(state)),
                     imread_unicode(state.images[0]).shape)
    path = tmp_path / name
    write_case(path, case, lambda i: imread_unicode(state.images[i]))
    return path


def _replay(path, **kw):
    with open_bundle(path) as bundle:
        return bundle.case, replay_case(bundle.case, bundle.read_frame, **kw)


@pytest.fixture
def edge_state(tmp_path):
    folder = write_sequence(tmp_path / "seq", RADII)
    return fit_all(folder_state(folder, roi=ROI, mode="edge"), reference=0)


class TestAutoMode:
    def test_the_crop_gives_the_full_frame_results(self, edge_state, tmp_path):
        case, replayed = _replay(_bundle(edge_state, tmp_path, reference=0))
        assert sorted(replayed) == [0, 1, 2, 3, 4, 5, 6, 7]
        for idx, now in replayed.items():
            app = case.frame(idx).result
            if app.radius is None:
                assert not now.accepted
                continue
            assert now.radius == app.radius
            assert now.center == app.center  # back in original-frame coordinates
            assert now.confidence == pytest.approx(edge_state.fit_confidence[idx])

    def test_tracking_replays_when_the_user_had_it_on(self, edge_state, tmp_path, monkeypatch):
        from bubbletrack.model import sequence

        calls = []
        real = sequence.solve_sequence

        def spy(raw, redetect, **kw):
            calls.append((len(raw), kw.get("anchors")))
            return real(raw, redetect, **kw)

        monkeypatch.setattr(sequence, "solve_sequence", spy)
        path = _bundle(edge_state, tmp_path, reference=0, tracking=True)
        _, replayed = _replay(path)
        assert calls == [(8, None)]  # every stored frame, never the labels as anchors
        assert all(replayed[i].accepted for i in range(1, 8))
        _replay(path, tracking=False)
        assert len(calls) == 1

    def test_a_tracked_remeasurement_replaces_the_frame(self, edge_state, tmp_path, monkeypatch):
        from bubbletrack.model import sequence
        from bubbletrack.model.sequence.types import FrameSolution, SequenceResult

        def fake(raw, redetect, **kw):
            obs = redetect(2, (raw[2].rc, raw[2].cc, raw[2].radius))
            frames = [FrameSolution(o.radius, o.rc, o.cc, o.confidence, "raw", frozenset(), 0)
                      for o in raw]
            frames[2] = FrameSolution(obs.radius, obs.rc, obs.cc, obs.confidence, "tracked",
                                      frozenset(), 0)
            return SequenceResult(frames=tuple(frames), collapse_frames=(), cycles=(), review=(),
                                  seed_frame=0, redetect_calls=1)

        monkeypatch.setattr(sequence, "solve_sequence", fake)
        _, replayed = _replay(_bundle(edge_state, tmp_path, reference=0, tracking=True))
        assert replayed[2].tracked and not replayed[3].tracked

    def test_seed_and_reference_are_moved_into_the_crop(self, edge_state, tmp_path, monkeypatch):
        state = dataclasses.replace(edge_state, edge_seed=(130.0, 210.0))
        path = _bundle(state, tmp_path, reference=0)
        seen = []

        def engine(raw, gridx, gridy, *, reference=None, params=None):
            seen.append((raw.shape, gridx, gridy, reference, params.seed))
            raise RuntimeError("stop")

        monkeypatch.setattr(edge_pipeline, "_engine", lambda: engine)
        with pytest.raises(RuntimeError):
            _replay(path)
        shape, gridx, gridy, reference, seed = seen[0]
        assert shape == (190, 280) and (gridx, gridy) == ((1, 190), (1, 280))
        full = imread_unicode(state.images[0])
        assert np.array_equal(reference, full[30:220, 60:340])
        assert seed == (100.0, 150.0)

    def test_max_radius_of_the_app_applies(self, edge_state, tmp_path):
        path = _bundle(edge_state, tmp_path, reference=0)
        with open_bundle(path) as bundle:
            det = dataclasses.replace(bundle.case.detection, max_radius=10.0)
            case = dataclasses.replace(bundle.case, detection=det)
            replayed = replay_case(case, bundle.read_frame)
        assert not any(r.accepted for r in replayed.values())
        assert replayed[5].reason == "implausible radius"

    def test_reference_missing_from_the_bundle(self, edge_state, tmp_path):
        path = _bundle(edge_state, tmp_path, reference=0)
        with open_bundle(path) as bundle:
            det = dataclasses.replace(bundle.case.detection, reference_index=99)
            case = dataclasses.replace(bundle.case, detection=det)
            with pytest.raises(CaseFormatError, match="reference frame 100"):
                replay_case(case, bundle.read_frame)


class TestThresholdModes:
    @pytest.mark.parametrize("mode", ["intensity", "gradient", "area"])
    def test_full_frame_roi_replays_exactly(self, tmp_path, mode):
        folder = write_sequence(tmp_path / "seq", RADII[1:])
        state = fit_all(folder_state(folder, mode=mode, **LEGACY))
        case, replayed = _replay(_bundle(state, tmp_path, start=2, end=3))
        fitted = 0
        for idx, now in replayed.items():
            app = case.frame(idx).result
            assert (now.radius, now.center) == (app.radius, app.center)
            fitted += app.radius is not None
        assert fitted >= 5

    def test_roi_crop_replays_close_to_the_app(self, tmp_path):
        folder = write_sequence(tmp_path / "seq", RADII[1:])
        state = fit_all(folder_state(folder, roi=ROI, **LEGACY))
        case, replayed = _replay(_bundle(state, tmp_path, start=2, end=3))
        for idx, now in replayed.items():
            app = case.frame(idx).result
            assert now.accepted and app.radius is not None
            assert now.radius == pytest.approx(app.radius, abs=1.0)

    def test_out_of_range_parameters_are_clamped(self, tmp_path):
        folder = write_sequence(tmp_path / "seq", RADII[1:4])
        state = fit_all(folder_state(folder, **LEGACY))
        path = _bundle(state, tmp_path, start=1, end=1)
        with open_bundle(path) as bundle:
            params = {**bundle.case.detection.params, "img_thr": 7.0, "removing_obj_radius": "x"}
            det = dataclasses.replace(bundle.case.detection, params=params)
            replayed = replay_case(dataclasses.replace(bundle.case, detection=det), bundle.read_frame)
        assert set(replayed) == {0, 1, 2}
