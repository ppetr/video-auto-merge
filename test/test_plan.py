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
"""Tests for plan (plain data structures, no jax or ffmpeg involved)."""

import numpy as np
import pytest

from video_auto_merge import plan
from video_auto_merge import report

FRAME_S = 0.5
LEFT = np.arange(20) * FRAME_S
RIGHT = np.arange(30) * FRAME_S + 0.1


def _both(ls, le, rs, re):
  return report.Block('both', ls, le, rs, re)


def test_identical_films_are_one_right_segment():
  segs = plan.BuildPlan([_both(0, 20, 0, 20)], LEFT, RIGHT, min_match_s=0)
  assert segs == [plan.Segment(plan.RIGHT, 0.0, 10.0, 0.1)]


def test_right_only_block_is_dropped_and_skipped_in_source():
  blocks = [
      _both(0, 8, 0, 8),
      report.Block('right', 8, 8, 8, 12),
      _both(8, 20, 12, 24),
  ]
  segs = plan.BuildPlan(blocks, LEFT, RIGHT, min_match_s=0)
  assert segs == [
      plan.Segment(plan.RIGHT, 0.0, 4.0, 0.1),
      plan.Segment(plan.RIGHT, 4.0, 10.0, 6.1),
  ]


def test_left_only_block_comes_from_fallback():
  blocks = [
      _both(0, 8, 0, 8),
      report.Block('left', 8, 12, 8, 8),
      _both(12, 20, 8, 16),
  ]
  segs = plan.BuildPlan(blocks, LEFT, RIGHT, min_match_s=0)
  assert segs == [
      plan.Segment(plan.RIGHT, 0.0, 4.0, 0.1),
      plan.Segment(plan.FALLBACK, 4.0, 6.0, 4.0),
      plan.Segment(plan.RIGHT, 6.0, 10.0, 4.1),
  ]


def test_short_match_between_gaps_merges_into_one_fallback():
  blocks = [
      report.Block('left', 0, 4, 0, 0),
      _both(4, 6, 0, 2),
      report.Block('left', 6, 10, 2, 2),
      _both(10, 20, 2, 12),
  ]
  segs = plan.BuildPlan(blocks, LEFT, RIGHT, min_match_s=2.0)
  assert segs == [
      plan.Segment(plan.FALLBACK, 0.0, 5.0, 0.0),
      plan.Segment(plan.RIGHT, 5.0, 10.0, 1.1),
  ]


def test_plan_covers_timeline_without_gaps():
  blocks = [
      _both(0, 5, 0, 5),
      report.Block('right', 5, 5, 5, 9),
      report.Block('left', 5, 7, 9, 9),
      _both(7, 20, 9, 22),
  ]
  segs = plan.BuildPlan(blocks, LEFT, RIGHT, min_match_s=0)
  assert segs[0].start == 0.0
  assert segs[-1].end == 10.0
  for prev, nxt in zip(segs, segs[1:]):
    assert prev.end == nxt.start


def test_plan_is_extended_back_to_time_zero():
  left = np.arange(20) * FRAME_S + 0.2
  segs = plan.BuildPlan([_both(0, 20, 0, 20)], left, RIGHT, min_match_s=0)
  assert segs[0].start == 0.0
  assert segs[0].source_start == pytest.approx(RIGHT[0] - 0.2)
  assert segs[0].end == pytest.approx(10.2)


def test_empty_blocks_give_empty_plan():
  assert not plan.BuildPlan([], LEFT, RIGHT)
