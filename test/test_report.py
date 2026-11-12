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
"""Tests for report (plain data structures, no jax or ffmpeg involved)."""

import dataclasses

import numpy as np
import pytest

from video_auto_merge import report


@dataclasses.dataclass
class _Aligned:
  left_start: int
  left_end: int
  right_start: int
  right_end: int


@dataclasses.dataclass
class _Unique:
  side: str
  start: int
  end: int


@dataclasses.dataclass
class _Result:
  aligned: list
  unique: list


def test_ordered_blocks_interleaves_matched_and_unique():
  result = _Result(aligned=[_Aligned(0, 10, 0, 10),
                            _Aligned(10, 20, 15, 25)],
                   unique=[_Unique('right', 10, 15)])

  blocks = report.OrderedBlocks(result)

  assert [b.kind for b in blocks] == ['both', 'right', 'both']
  assert (blocks[1].left_start, blocks[1].left_end) == (10, 10)
  assert (blocks[1].right_start, blocks[1].right_end) == (10, 15)


def test_ordered_blocks_starts_and_ends_with_unique_blocks():
  result = _Result(aligned=[_Aligned(3, 8, 0, 5)],
                   unique=[_Unique('left', 0, 3),
                           _Unique('right', 5, 9)])

  blocks = report.OrderedBlocks(result)

  assert [b.kind for b in blocks] == ['left', 'both', 'right']
  assert blocks[-1].left_start == 8


def test_ordered_blocks_rejects_inconsistent_input():
  result = _Result(aligned=[_Aligned(5, 8, 5, 8)], unique=[])
  with pytest.raises(ValueError):
    report.OrderedBlocks(result)


def test_output_frames_counts_matched_frames_once():
  blocks = [
      report.Block('both', 0, 10, 0, 10),
      report.Block('right', 10, 10, 10, 15),
      report.Block('left', 10, 14, 15, 15)
  ]
  assert [b.output_frames for b in blocks] == [10, 5, 4]


def test_format_table_orders_rows_and_can_hide_matches():
  times = np.arange(30) * 0.5
  blocks = [
      report.Block('both', 0, 10, 0, 10),
      report.Block('right', 10, 10, 10, 15),
      report.Block('both', 10, 20, 15, 25)
  ]

  full = report.FormatTable(blocks, times, times)
  diff = report.FormatTable(blocks, times, times, only_differences=True)

  assert len(full.splitlines()) == 4
  assert len(diff.splitlines()) == 2
  assert '0:00:05.000' in diff  # out start of the only unique block.
  assert diff.splitlines()[1].split()[1] == 'right'
