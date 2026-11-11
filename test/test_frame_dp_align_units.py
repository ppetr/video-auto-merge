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


@pytest.mark.parametrize('c0,c1', [(5.0, 1.0), (20.0, 3.0), (1.0, 0.5)])
def test_backtrack_matches_naive_optimum(c0, c1):
  rng = np.random.RandomState(2)
  for _ in range(10):
    n = rng.randint(5, 60)
    m = rng.randint(5, 60)
    D = _L1Cost(_RandomSignatures(rng, n), _RandomSignatures(rng, m))
    true_total = frame_dp_align.NaiveFrameDp(D, c0, c1)[n, m]

    path = frame_dp_align.Backtrack(D, c0, c1, base_size=4)
    ok, msg = frame_dp_align.PathEndpoints(path, n, m)
    assert ok, msg
    got_cost = frame_dp_align.PathCost(path, D, c0, c1)
    assert got_cost == pytest.approx(true_total, abs=1e-2)


def test_backtrack_exercises_in_flight_split():
  # A long run of one-sided content, placed so that Hirschberg's split
  # row necessarily falls inside the jump (not at either endpoint),
  # forces the in-flight-jump correction's branch to be taken at least
  # once - without it, the split would double-charge the jump's flat
  # c0 cost and this assertion would catch the resulting cost mismatch.
  # The extra block is on the *left* side (so recovering it is an
  # i-jump) and placed so it straddles the midpoint of the full left
  # range - Hirschberg's top-level split row necessarily falls inside
  # it, not at either endpoint.
  rng = np.random.RandomState(3)
  sig = _RandomSignatures(rng, 20)
  extra = _RandomSignatures(rng, 30)
  left_sig = np.concatenate([sig[:10], extra, sig[10:]])
  D = _L1Cost(left_sig, sig)
  c0, c1 = 5.0, 0.1
  true_total = frame_dp_align.NaiveFrameDp(D, c0, c1)[50, 20]

  frame_dp_align.SPLIT_STATS['node'] = 0
  frame_dp_align.SPLIT_STATS['in_flight'] = 0
  path = frame_dp_align.Backtrack(D, c0, c1, base_size=4)
  ok, msg = frame_dp_align.PathEndpoints(path, 50, 20)
  assert ok, msg
  got_cost = frame_dp_align.PathCost(path, D, c0, c1)
  assert got_cost == pytest.approx(true_total, abs=1e-2)
  assert frame_dp_align.SPLIT_STATS['in_flight'] > 0
