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
"""Tests for subtitles (plain data, no ffmpeg)."""

import pytest

from video_auto_merge import plan
from video_auto_merge import subtitles

SRT = """1
00:00:01,000 --> 00:00:02,500
Hello
world

2
00:01:00,250 --> 00:01:01,000
Bye
"""


def test_parse_and_format_roundtrip():
  cues = subtitles.ParseSrt(SRT)
  assert cues == [
      subtitles.Cue(1.0, 2.5, 'Hello\nworld'),
      subtitles.Cue(60.25, 61.0, 'Bye')
  ]
  assert subtitles.ParseSrt(subtitles.FormatSrt(cues)) == cues


def test_parse_accepts_crlf_and_dot_milliseconds():
  cues = subtitles.ParseSrt('1\r\n00:00:01.5 --> 00:00:02.000\r\nHi\r\n')
  assert cues == [subtitles.Cue(1.5, 2.0, 'Hi')]


def _right(start, end, source_start):
  return plan.Segment(plan.RIGHT, start, end, source_start)


def test_cue_inside_segment_is_shifted():
  cues = [subtitles.Cue(11.0, 12.0, 'a')]
  out = subtitles.RemapCues(cues, [_right(0.0, 10.0, 10.0)])
  assert out == [subtitles.Cue(1.0, 2.0, 'a')]


def test_cue_in_dropped_stretch_disappears():
  cues = [subtitles.Cue(5.0, 6.0, 'a')]
  segs = [_right(0.0, 2.0, 0.0), _right(2.0, 4.0, 10.0)]
  assert not subtitles.RemapCues(cues, segs)


def test_cue_crossing_a_cut_is_split():
  cues = [subtitles.Cue(1.0, 11.0, 'a')]
  segs = [_right(0.0, 2.0, 0.0), _right(2.0, 6.0, 10.0)]
  assert subtitles.RemapCues(cues, segs) == [
      subtitles.Cue(1.0, 2.0, 'a'),
      subtitles.Cue(2.0, 3.0, 'a'),
  ]


def test_fallback_segments_carry_no_cues():
  cues = [subtitles.Cue(1.0, 2.0, 'a')]
  segs = [plan.Segment(plan.FALLBACK, 0.0, 5.0, 0.0)]
  assert not subtitles.RemapCues(cues, segs)


def test_tiny_fragments_are_dropped_but_short_cues_survive():
  segs = [_right(0.0, 1.0, 0.0)]
  long_cue_edge = subtitles.Cue(0.95, 5.5, 'x')
  short_cue = subtitles.Cue(0.5, 0.55, 'y')
  clipped_short_cue = subtitles.Cue(0.98, 1.03, 'z')
  assert subtitles.RemapCues([long_cue_edge, clipped_short_cue], segs) == []
  assert subtitles.RemapCues([short_cue], segs) == [short_cue]


def test_output_is_sorted():
  cues = [subtitles.Cue(10.5, 11.0, 'b'), subtitles.Cue(0.2, 0.9, 'a')]
  segs = [_right(0.0, 1.0, 0.0), _right(1.0, 2.0, 10.0)]
  out = subtitles.RemapCues(cues, segs)
  assert [c.text for c in out] == ['a', 'b']
  assert out[1].start == pytest.approx(1.5)
