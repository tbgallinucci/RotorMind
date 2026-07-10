"""Dimensionless unit-consistency checks for the solver output.

Each ratio pits a solver result against an independent closed-form value with
the same physical dimension, so units cancel. A meters/micrometers mix-up
anywhere in the chain would blow these ratios up by 1e3-1e6.
"""

import numpy as np
import pytest

from engine.rotordynamics.analysis import RotordynamicAnalysis


@pytest.fixture(scope="module")
def analysis():
    a = RotordynamicAnalysis()
    a.omega = np.arange(10, 2001, 50)   # coarse but spans sub- to supercritical
    a.n = len(a.omega)
    a.initialize_arrays()
    assert a.run_analysis() is True
    return a


def test_static_sag_matches_beam_theory(analysis):
    """Quasi-static Z deflection at the disk ~ simply-supported beam formula."""
    a = analysis
    delta_solver = abs(a.X[22, 0])                      # lowest speed ~ static
    aa, L = a.d1, a.d2
    bb = L - aa
    delta_beam = a.W * aa**2 * bb**2 / (3 * a.E * a.Ie * L)
    assert 0.5 < delta_solver / delta_beam < 2.0


def test_effective_stiffness_matches_beam_theory(analysis):
    a = analysis
    k_solver = a.W / abs(a.X[22, 0])
    k_beam = 3 * a.E * a.Ie * a.d2 / (a.d1**2 * (a.d2 - a.d1)**2)
    assert 0.5 < k_solver / k_beam < 2.0


def test_first_critical_consistent_with_sqrt_k_over_m(analysis):
    """w1 from the eigenproblem vs sqrt(k/m) from the response amplitude.
    A um-vs-m displacement error would skew this ratio by ~1e3."""
    a = analysis
    k_solver = a.W / abs(a.X[22, 0])
    w1_est = np.sqrt(k_solver / a.md)
    ratio = a.critical_speeds[0] / w1_est
    assert 0.5 < ratio < 1.5


def test_supercritical_amplitude_approaches_eccentricity(analysis):
    """Jeffcott limit: far above the critical, |Y_disk| -> e = m.e/m exactly.
    This is the sharpest unit fingerprint in the model."""
    a = analysis
    e = a.me / a.md
    ratio = abs(a.X[20, -1]) / e
    assert 0.5 < ratio < 1.5
