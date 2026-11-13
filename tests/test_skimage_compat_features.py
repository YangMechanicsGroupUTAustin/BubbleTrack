"""ML assist features work on scikit-image versions with and without num_workers."""

from __future__ import annotations

import numpy as np

from bubbletrack.model.skimage_compat import multiscale_basic_features_compat


def test_num_workers_is_dropped_where_unsupported():
    """scikit-image 0.26 removed the argument (owner's Mac: "unexpected keyword argument")."""
    seen = {}

    def new_api(image, intensity=True, edges=True, texture=True, sigma_min=0.5, sigma_max=16,
                num_sigma=None, *, channel_axis=None):
        seen.update(intensity=intensity, sigma_max=sigma_max)
        return np.zeros(image.shape + (1,))

    out = multiscale_basic_features_compat(np.zeros((8, 8)), num_workers=1, sigma_max=4, func=new_api)
    assert out.shape == (8, 8, 1) and seen == {"intensity": True, "sigma_max": 4}


def test_num_workers_is_passed_where_supported():
    seen = {}

    def old_api(image, intensity=True, num_workers=None):
        seen["num_workers"] = num_workers
        return np.zeros(image.shape + (1,))

    multiscale_basic_features_compat(np.zeros((8, 8)), num_workers=3, func=old_api)
    assert seen == {"num_workers": 3}


def test_installed_version_runs():
    out = multiscale_basic_features_compat(np.random.default_rng(0).random((20, 20)), num_workers=1,
                                           sigma_max=4)
    assert out.shape[:2] == (20, 20)
