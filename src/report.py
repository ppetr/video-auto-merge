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
"""Human-readable report of a `frame_dp_align.AlignFrames` result.

`AlignFrames` returns matched and unique blocks in two separate lists.
`OrderedBlocks` merges them back into one sequence in the order of the
optimal path, i.e. the order in which the blocks would appear in a film
assembled from both inputs; `FormatTable` renders that sequence as a text
table. Works on any object with `aligned`/`unique` block lists shaped like
`frame_dp_align.FrameAlignmentResult` (no jax needed).
"""
import dataclasses
from typing import List

import numpy as np
import numpy.typing as npt


@dataclasses.dataclass
class Block:
  """One block of the path. `kind` is 'both' (frames matched 1:1), 'left'
  (only in the left input) or 'right' (only in the right input). Ranges are
  `[start, end)` frame indices; for a unique block the other side's range
  is empty and marks the position in that input where the block sits."""
  kind: str
  left_start: int
  left_end: int
  right_start: int
  right_end: int

  @property
  def output_frames(self) -> int:
    """How many frames the block takes in the merged film."""
    return max(self.left_end - self.left_start,
               self.right_end - self.right_start)


def OrderedBlocks(result) -> List[Block]:
  """Merges `result.aligned` and `result.unique` into path order."""
  aligned = list(result.aligned)
  unique = list(result.unique)
  blocks = []
  i = j = 0
  a = u_left = u_right = 0
  lefts = [u for u in unique if u.side == 'left']
  rights = [u for u in unique if u.side == 'right']
  while a < len(aligned) or u_left < len(lefts) or u_right < len(rights):
    if (a < len(aligned) and aligned[a].left_start == i and
        aligned[a].right_start == j):
      blk = aligned[a]
      blocks.append(
          Block('both', blk.left_start, blk.left_end, blk.right_start,
                blk.right_end))
      i, j = blk.left_end, blk.right_end
      a += 1
    elif u_left < len(lefts) and lefts[u_left].start == i:
      blk = lefts[u_left]
      blocks.append(Block('left', blk.start, blk.end, j, j))
      i = blk.end
      u_left += 1
    elif u_right < len(rights) and rights[u_right].start == j:
      blk = rights[u_right]
      blocks.append(Block('right', i, i, blk.start, blk.end))
      j = blk.end
      u_right += 1
    else:
      raise ValueError(f'blocks are not a consistent path at ({i}, {j})')
  return blocks


ABSENT = '-'
EMPTY = ''
COLUMNS = ('#', 'kind', 'out', 'left', 'frames', 'right')


def _Timestamp(seconds: float) -> str:
  millis = int(round(seconds * 1000))
  sign = '-' if millis < 0 else ''
  millis = abs(millis)
  return (f'{sign}{millis // 3600000:d}:{millis // 60000 % 60:02d}:'
          f'{millis // 1000 % 60:02d}.{millis % 1000:03d}')


def FormatTable(blocks: List[Block],
                left_times: npt.ArrayLike,
                right_times: npt.ArrayLike,
                only_differences: bool = False) -> str:
  """Renders `blocks` as a text table, one row per block in path order.

  `out` is the block's start in the merged film (every frame counted once,
  matched frames included), `left`/`right` the block's time range in each
  input (`-` where the block is absent from it). With `only_differences`,
  matched blocks are omitted (their `out` still accounts for them).
  """
  left_times = np.asarray(left_times)
  right_times = np.asarray(right_times)
  frame_s = float((left_times[-1] - left_times[0]) / (len(left_times) - 1))

  def Span(times, start, end):
    if start == end:
      return f'{ABSENT:>25}  {EMPTY:>7}'
    stop = times[end] if end < len(times) else times[-1] + frame_s
    return (f'{_Timestamp(times[start])} - {_Timestamp(stop)}  '
            f'{end - start:>7d}')

  lines = [
      f'{COLUMNS[0]:>4}  {COLUMNS[1]:<5}  {COLUMNS[2]:>12}  '
      f'{COLUMNS[3]:>25}  {COLUMNS[4]:>7}  {COLUMNS[5]:>25}  {COLUMNS[4]:>7}'
  ]
  out_frames = 0
  for n, blk in enumerate(blocks):
    if not only_differences or blk.kind != 'both':
      lines.append(
          f'{n:>4}  {blk.kind:<5}  {_Timestamp(out_frames * frame_s):>12}  '
          f'{Span(left_times, blk.left_start, blk.left_end)}  '
          f'{Span(right_times, blk.right_start, blk.right_end)}')
    out_frames += blk.output_frames
  return '\n'.join(lines) + '\n'
