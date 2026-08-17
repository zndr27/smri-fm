"""CPU tests for the task 2 protocol. The encoder cannot run on CPU, so nothing here builds one:
`binarize` and `subject_curves` only read `cfg`, so a stand-in stands in for the method."""

from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from fomo_tune.main_task2 import (
    THRESHOLDS,
    Config,
    Curves,
    Task2Method,
    normalized_surface_distance,
    score,
    subject_curves,
)


@pytest.fixture
def method():
    stand_in = SimpleNamespace(cfg=Config(largest_component=False))
    stand_in.binarize = MethodType(Task2Method.binarize, stand_in)
    return stand_in


def cube(shift=0):
    mask = np.zeros((40, 40, 40), dtype=bool)
    mask[10 + shift : 30 + shift, 10:30, 10:30] = True
    return mask


def test_thresholds_contain_the_old_grid():
    """The grid was extended to reach 1, so every score already recorded stays reachable."""
    old = np.logspace(-6, -0.3, 60)
    assert np.isin(old, THRESHOLDS).all()
    assert THRESHOLDS.max() > 0.99


def test_nsd_identical_masks_score_one():
    mask = cube()
    assert normalized_surface_distance(mask, mask, (1.0, 1.0, 1.0)) == 1.0


def test_nsd_disjoint_masks_score_zero():
    far = np.zeros((40, 40, 40), dtype=bool)
    far[0:5, 0:5, 0:5] = True
    assert normalized_surface_distance(far, cube(), (1.0, 1.0, 1.0)) == 0.0


def test_nsd_scores_zero_when_either_mask_is_empty():
    """The challenge's wrapper scores an empty mask 0, where its Dice would score 1."""
    empty = np.zeros((40, 40, 40), dtype=bool)
    assert normalized_surface_distance(empty, cube(), (1.0, 1.0, 1.0)) == 0.0
    assert normalized_surface_distance(cube(), empty, (1.0, 1.0, 1.0)) == 0.0
    assert normalized_surface_distance(empty, empty, (1.0, 1.0, 1.0)) == 0.0


def test_nsd_falls_off_beyond_the_tolerance():
    """A 1mm shift is exactly within the 1mm tolerance, so it still scores 1; further does not."""
    truth = cube()
    assert normalized_surface_distance(cube(shift=1), truth, (1.0, 1.0, 1.0)) == 1.0

    drifting = [
        normalized_surface_distance(cube(shift=s), truth, (1.0, 1.0, 1.0)) for s in (2, 4, 8)
    ]
    assert drifting == sorted(drifting, reverse=True)
    assert drifting[-1] < drifting[0] < 1.0


def test_nsd_depends_on_spacing():
    """Slices here are 5-7mm against sub-mm in plane, so passing (1,1,1) would flatter the score."""
    truth, prediction = cube(), cube(shift=2)
    isotropic = normalized_surface_distance(prediction, truth, (1.0, 1.0, 1.0))
    anisotropic = normalized_surface_distance(prediction, truth, (6.0, 1.0, 1.0))
    assert isotropic != anisotropic


def test_subject_curves_agree_at_the_extremes(method):
    """Everything on at the lowest threshold, nothing on above the highest probability."""
    truth = cube()
    probabilities = np.where(truth, 0.9, 1e-3)

    dice, nsd, predicted = subject_curves(method, probabilities, truth, (1.0, 1.0, 1.0))

    assert dice.shape == nsd.shape == predicted.shape == THRESHOLDS.shape
    assert predicted[0] == truth.size
    assert predicted[-1] == 0
    assert nsd[-1] == 0.0
    best = dice.argmax()
    assert dice[best] == 1.0 and nsd[best] == 1.0


def test_score_reports_both_metrics_and_ships_the_dice_cut():
    n_subjects, n_thresholds = 5, len(THRESHOLDS)
    dice = np.zeros((n_subjects, n_thresholds))
    nsd = np.zeros((n_subjects, n_thresholds))
    dice[:, 10] = 0.4
    nsd[:, 20] = 0.3

    summary = score(Curves(dice, nsd, np.zeros((n_subjects, n_thresholds)), np.ones(n_subjects)))

    assert summary["dice"] == pytest.approx(0.4)
    assert summary["nsd"] == pytest.approx(0.3)
    assert summary["dice_threshold"] == THRESHOLDS[10]
    assert summary["nsd_threshold"] == THRESHOLDS[20]
    assert summary["threshold"] == summary["dice_threshold"]
