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
"""Turns an alignment into a plan for assembling a stream on the left timeline.

The output of a merge has the left film's timeline. Each `Segment` covers
a part of it and says where its content comes from: the right film (a
matched block), or a fall-back stream of the left film (a block that has
no counterpart in the right film). Blocks present only in the right film
do not appear in the plan at all - they are dropped.
"""
import dataclasses
from typing import List

import numpy as np
import numpy.typing as npt

from video_auto_merge import report

RIGHT = 'right'
FALLBACK = 'fallback'

DEFAULT_MIN_MATCH_S = 0.5


@dataclasses.dataclass
class Segment:
  """`[start, end)` of the left timeline (seconds). `source` is RIGHT or
  FALLBACK; `source_start` is the time in the source stream that lands on
  `start` (the same length is taken, there is no speed change). For
  FALLBACK the source is on the left timeline too, so `source_start ==
  start`."""
  source: str
  start: float
  end: float
  source_start: float

  @property
  def duration(self) -> float:
    return self.end - self.start


def BuildPlan(blocks: List[report.Block],
              left_times: npt.ArrayLike,
              right_times: npt.ArrayLike,
              min_match_s: float = DEFAULT_MIN_MATCH_S) -> List[Segment]:
  """Builds contiguous segments covering the whole left timeline.

  The plan starts at time 0 (or at the first left frame if that is earlier)
  and ends after the last left frame. Matched blocks shorter than
  `min_match_s` are taken from the fall-back instead, since they cannot be
  crossfaded sensibly between two neighbouring cuts.
  """
  left_times = np.asarray(left_times, dtype=float)
  right_times = np.asarray(right_times, dtype=float)
  frame_s = float((left_times[-1] - left_times[0]) / (len(left_times) - 1))

  def LeftTime(index: int) -> float:
    if index < len(left_times):
      return float(left_times[index])
    return float(left_times[-1]) + frame_s

  segments: List[Segment] = []
  for blk in blocks:
    if blk.kind == 'right':
      continue
    start = LeftTime(blk.left_start)
    end = LeftTime(blk.left_end)
    if blk.kind == 'both' and end - start >= min_match_s:
      segments.append(
          Segment(RIGHT, start, end, float(right_times[blk.right_start])))
    elif segments and segments[-1].source == FALLBACK:
      segments[-1].end = end
    else:
      segments.append(Segment(FALLBACK, start, end, start))

  if segments:
    first = segments[0]
    origin = min(0.0, float(left_times[0]))
    first.source_start -= first.start - origin
    first.start = origin
  return segments
