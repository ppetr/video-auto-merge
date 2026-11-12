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
"""Re-times SRT subtitles of the right video onto the left timeline.

Cues are cut along the RIGHT segments of a plan (see `plan`): the part of a
cue inside a segment moves with that segment, parts in dropped or
fall-back stretches disappear.
"""
import dataclasses
import re
from typing import List

from video_auto_merge import plan

DEFAULT_MIN_CUE_S = 0.1

_TIME = r'(\d+):(\d\d):(\d\d)[,.](\d{1,3})'
_TIMING = re.compile(rf'^\s*{_TIME}\s*-->\s*{_TIME}')


@dataclasses.dataclass
class Cue:
  start: float
  end: float
  text: str


def _Seconds(hours: str, minutes: str, seconds: str, millis: str) -> float:
  return (int(hours) * 3600 + int(minutes) * 60 + int(seconds) +
          int(millis.ljust(3, '0')) / 1000)


def ParseSrt(text: str) -> List[Cue]:
  """Parses SRT text; blocks without a timing line are ignored."""
  cues = []
  for block in re.split(r'\n\s*\n', text.replace('\r\n', '\n').strip()):
    lines = block.split('\n')
    for n, line in enumerate(lines):
      match = _TIMING.match(line)
      if match:
        cues.append(
            Cue(_Seconds(*match.groups()[:4]), _Seconds(*match.groups()[4:]),
                '\n'.join(lines[n + 1:])))
        break
  return cues


def _Timestamp(seconds: float) -> str:
  millis = round(seconds * 1000)
  return (f'{millis // 3600000:02d}:{millis // 60000 % 60:02d}:'
          f'{millis // 1000 % 60:02d},{millis % 1000:03d}')


def FormatSrt(cues: List[Cue]) -> str:
  return ''.join(f'{n}\n{_Timestamp(c.start)} --> {_Timestamp(c.end)}\n'
                 f'{c.text}\n\n' for n, c in enumerate(cues, 1))


def RemapCues(cues: List[Cue],
              segments: List[plan.Segment],
              min_cue_s: float = DEFAULT_MIN_CUE_S) -> List[Cue]:
  """Moves cues onto the left timeline, in order. A cue spanning several
  segments is split; pieces shorter than `min_cue_s` are dropped."""
  moved = []
  for seg in segments:
    if seg.source != plan.RIGHT:
      continue
    source_end = seg.source_start + seg.duration
    shift = seg.start - seg.source_start
    for cue in cues:
      start = max(cue.start, seg.source_start)
      end = min(cue.end, source_end)
      if end > start and end - start >= min(min_cue_s, cue.end - cue.start):
        moved.append(Cue(start + shift, end + shift, cue.text))
  return sorted(moved, key=lambda c: (c.start, c.end))
