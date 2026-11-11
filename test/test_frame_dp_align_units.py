#!/usr/bin/env python3

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for frame_dp_align. Builds small synthetic ternary "signature"
arrays directly in numpy (no ffmpeg involved - frame_dp_align operates on
already-extracted per-frame descriptors, so synthetic descriptor arrays
are a faithful test input, not a simplification of the real one)."""

# pylint: disable=missing-function-docstring,missing-class-docstring

import numpy as np
import pytest

import frame_dp_align

_NDIM = 12  # small, like the real MPEG-7 signature's ternary descriptor.


def _RandomSignatures(rng, n, ndim=_NDIM):
  return rng.randint(0, 3, size=(n, ndim))


def _L1Cost(left_sig, right_sig):
  return np.abs(left_sig[:, None, :] - right_sig[None, :, :]).sum(
      axis=-1).astype(np.float64)


@pytest.mark.parametrize('c0,c1', [(5.0, 1.0), (20.0, 3.0), (1.0, 0.5)])
def test_fast_matches_naive(c0, c1):
  rng = np.random.RandomState(0)
  for _ in range(6):
    n = rng.randint(3, 15)
    m = rng.randint(3, 15)
    D = _L1Cost(_RandomSignatures(rng, n), _RandomSignatures(rng, m))
    naive = frame_dp_align.NaiveFrameDp(D, c0, c1)
    fast = frame_dp_align.FastFrameDp(D, c0, c1)
    np.testing.assert_allclose(fast, naive, atol=1e-6)


def test_base_case():
  D = np.zeros((0, 0))
  assert frame_dp_align.NaiveFrameDp(D, 5.0, 1.0)[0, 0] == 0.0
  assert frame_dp_align.FastFrameDp(D, 5.0, 1.0)[0, 0] == 0.0


def test_rejects_negative_params():
  D = np.zeros((2, 2))
  with pytest.raises(ValueError):
    frame_dp_align.NaiveFrameDp(D, -1.0, 1.0)
  with pytest.raises(ValueError):
    frame_dp_align.FastFrameDp(D, 1.0, -1.0)


def test_identical_sequences_prefer_matching():
  # Identical descriptors on both sides: an all-diagonal path (cost 0)
  # must beat any path involving a jump (cost >= c0 > 0).
  rng = np.random.RandomState(1)
  sig = _RandomSignatures(rng, 10)
  D = _L1Cost(sig, sig)
  F = frame_dp_align.FastFrameDp(D, c0=5.0, c1=1.0)
  assert F[10, 10] == 0.0
