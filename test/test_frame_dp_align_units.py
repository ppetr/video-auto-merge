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

jax = pytest.importorskip('jax')  # frame_dp_align imports jax at module
# scope for its GPU path, so the whole file (including the CPU-only
# tests above) needs it to even import the module - skip together
# rather than trying to make the CPU tests independently importable.

from video_auto_merge import frame_dp_align  # noqa: E402  pylint: disable=wrong-import-position

_NDIM = 12  # small, like the real MPEG-7 signature's ternary descriptor.


def _random_signatures(rng, n, ndim=_NDIM):
  return rng.randint(0, 3, size=(n, ndim))


def _l1_cost(left_sig, right_sig):
  return np.abs(left_sig[:, None, :] -
                right_sig[None, :, :]).sum(axis=-1).astype(np.float64)


@pytest.mark.parametrize('c0,c1', [(5.0, 1.0), (20.0, 3.0), (1.0, 0.5)])
def test_fast_matches_naive(c0, c1):
  rng = np.random.RandomState(0)
  for _ in range(6):
    n = rng.randint(3, 15)
    m = rng.randint(3, 15)
    cost = _l1_cost(_random_signatures(rng, n), _random_signatures(rng, m))
    naive = frame_dp_align.NaiveFrameDp(cost, c0, c1)
    fast = frame_dp_align.FastFrameDp(cost, c0, c1)
    np.testing.assert_allclose(fast, naive, atol=1e-6)


def test_base_case():
  cost = np.zeros((0, 0))
  assert frame_dp_align.NaiveFrameDp(cost, 5.0, 1.0)[0, 0] == 0.0
  assert frame_dp_align.FastFrameDp(cost, 5.0, 1.0)[0, 0] == 0.0


def test_rejects_negative_params():
  cost = np.zeros((2, 2))
  with pytest.raises(ValueError):
    frame_dp_align.NaiveFrameDp(cost, -1.0, 1.0)
  with pytest.raises(ValueError):
    frame_dp_align.FastFrameDp(cost, 1.0, -1.0)


def test_identical_sequences_prefer_matching():
  # Identical descriptors on both sides: an all-diagonal path (cost 0)
  # must beat any path involving a jump (cost >= c0 > 0).
  rng = np.random.RandomState(1)
  sig = _random_signatures(rng, 10)
  cost = _l1_cost(sig, sig)
  cost_to = frame_dp_align.FastFrameDp(cost, c0=5.0, c1=1.0)
  assert cost_to[10, 10] == 0.0


@pytest.mark.parametrize('c0,c1', [(5.0, 1.0), (20.0, 3.0), (1.0, 0.5)])
def test_backtrack_matches_naive_optimum(c0, c1):
  rng = np.random.RandomState(2)
  for _ in range(10):
    n = rng.randint(5, 60)
    m = rng.randint(5, 60)
    cost = _l1_cost(_random_signatures(rng, n), _random_signatures(rng, m))
    true_total = frame_dp_align.NaiveFrameDp(cost, c0, c1)[n, m]

    path = frame_dp_align.Backtrack(cost, c0, c1, base_size=4)
    ok, msg = frame_dp_align.PathEndpoints(path, n, m)
    assert ok, msg
    got_cost = frame_dp_align.PathCost(path, cost, c0, c1)
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
  sig = _random_signatures(rng, 20)
  extra = _random_signatures(rng, 30)
  left_sig = np.concatenate([sig[:10], extra, sig[10:]])
  cost = _l1_cost(left_sig, sig)
  c0, c1 = 5.0, 0.1
  true_total = frame_dp_align.NaiveFrameDp(cost, c0, c1)[50, 20]

  frame_dp_align.SPLIT_STATS['node'] = 0
  frame_dp_align.SPLIT_STATS['in_flight'] = 0
  path = frame_dp_align.Backtrack(cost, c0, c1, base_size=4)
  ok, msg = frame_dp_align.PathEndpoints(path, 50, 20)
  assert ok, msg
  got_cost = frame_dp_align.PathCost(path, cost, c0, c1)
  assert got_cost == pytest.approx(true_total, abs=1e-2)
  assert frame_dp_align.SPLIT_STATS['in_flight'] > 0


def _one_hot_ternary(sig):
  return np.eye(3, dtype=np.float32)[sig].reshape(sig.shape[0], -1)


def _cost_transform_ternary(sig):
  cost_table = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=np.float32)
  return cost_table[sig].reshape(sig.shape[0], -1)


@pytest.mark.parametrize('c0,c1', [(5.0, 1.0), (20.0, 3.0), (1.0, 0.5)])
def test_streaming_backtrack_matches_naive_optimum(c0, c1):
  rng = np.random.RandomState(4)
  for _ in range(6):
    n = rng.randint(5, 60)
    m = rng.randint(5, 60)
    left_sig = _random_signatures(rng, n)
    right_sig = _random_signatures(rng, m)
    cost = _l1_cost(left_sig, right_sig)
    true_total = frame_dp_align.NaiveFrameDp(cost, c0, c1)[n, m]

    left_onehot = _one_hot_ternary(left_sig)
    right_cost = _cost_transform_ternary(right_sig)
    left_onehot_rev = _one_hot_ternary(left_sig[::-1])
    right_cost_rev = _cost_transform_ternary(right_sig[::-1])

    path = frame_dp_align.StreamingBacktrack(left_onehot,
                                             right_cost,
                                             left_onehot_rev,
                                             right_cost_rev,
                                             c0,
                                             c1,
                                             n,
                                             m,
                                             0,
                                             n,
                                             0,
                                             m,
                                             base_size=4,
                                             chunk=3)
    ok, msg = frame_dp_align.PathEndpoints(path, n, m)
    assert ok, msg
    got_cost = frame_dp_align.PathCost(path, cost, c0, c1)
    assert got_cost == pytest.approx(true_total, abs=1e-2)


def test_align_frames_finds_a_right_side_insert():
  rng = np.random.RandomState(5)
  left_sig = _random_signatures(rng, 30)
  extra = _random_signatures(rng, 15)
  right_sig = np.concatenate([left_sig[:10], extra, left_sig[10:]])

  params = frame_dp_align.FrameDpParams(c0=5.0, c1=0.1, base_size=4, chunk=3)
  result = frame_dp_align.AlignFrames(left_sig, right_sig, params)

  assert result.unique == [frame_dp_align.UniqueBlock('right', 10, 25)]
  assert result.aligned == [
      frame_dp_align.AlignedBlock(0, 10, 0, 10),
      frame_dp_align.AlignedBlock(10, 30, 25, 45),
  ]


def test_one_hot_cost_transform_gemm_matches_l1():
  rng = np.random.RandomState(6)
  left_sig = _random_signatures(rng, 8)
  right_sig = _random_signatures(rng, 5)
  expected = _l1_cost(left_sig, right_sig)
  got = frame_dp_align.OneHot(left_sig) @ frame_dp_align.CostTransform(
      right_sig).T
  np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize('c0,c1', [(5.0, 1.0), (20.0, 3.0), (1.0, 0.5)])
@pytest.mark.parametrize('shape', [(0, 7), (7, 0), (1, 1), (12, 9), (9, 30),
                                   (40, 5)])
def test_fast_base_case_is_optimal(c0, c1, shape):
  rng = np.random.RandomState(7)
  n, m = shape
  cost = _l1_cost(_random_signatures(rng, n), _random_signatures(rng, m))
  path = frame_dp_align._FastBaseCase(cost, c0, c1, 0, 0)  # pylint: disable=protected-access
  ok, msg = frame_dp_align.PathEndpoints(path, n, m)
  assert ok, msg
  assert frame_dp_align.PathCost(path, cost, c0, c1) == pytest.approx(
      frame_dp_align.NaiveFrameDp(cost, c0, c1)[n, m], abs=1e-6)


def test_fast_base_case_applies_offsets():
  cost = np.zeros((3, 3))
  path = frame_dp_align._FastBaseCase(cost, 5.0, 1.0, 10, 20)  # pylint: disable=protected-access
  assert path[0][1:3] == (10, 20)
  assert path[-1][3:5] == (13, 23)
