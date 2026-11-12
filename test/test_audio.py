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
"""Tests for audio (numpy only, no ffmpeg)."""

import numpy as np
import pytest

from video_auto_merge import audio
from video_auto_merge import plan

RATE = 100


def _ramp(n):
  return audio.ArraySource(np.arange(n, dtype=np.float32)[:, None])


def _const(n, value, channels=1):
  return audio.ArraySource(np.full((n, channels), value, dtype=np.float32))


def _assemble(segments, right, fallback, **kwargs):
  chunks = list(audio.AssembleAudio(segments, right, fallback, RATE, **kwargs))
  return np.concatenate(chunks)[:, 0]


def test_single_segment_is_shifted_source():
  segs = [plan.Segment(plan.RIGHT, 0.0, 2.0, 1.0)]
  out = _assemble(segs, _ramp(1000), _const(1000, 0.0))
  np.testing.assert_array_equal(out, 100 + np.arange(200))


def test_output_length_is_plan_length_with_crossfades():
  segs = [
      plan.Segment(plan.RIGHT, 0.0, 1.0, 0.0),
      plan.Segment(plan.FALLBACK, 1.0, 1.5, 1.0),
      plan.Segment(plan.RIGHT, 1.5, 3.0, 2.0),
  ]
  out = _assemble(segs, _const(1000, 1.0), _const(1000, 2.0))
  assert len(out) == 300


def test_values_away_from_joins_are_untouched():
  segs = [
      plan.Segment(plan.RIGHT, 0.0, 1.0, 0.0),
      plan.Segment(plan.FALLBACK, 1.0, 2.0, 1.0),
  ]
  out = _assemble(segs, _const(1000, 1.0), _const(1000, 2.0), crossfade_s=0.1)
  np.testing.assert_array_equal(out[:95], 1.0)
  np.testing.assert_array_equal(out[105:], 2.0)


def test_crossfade_is_equal_power_and_covers_join():
  segs = [
      plan.Segment(plan.RIGHT, 0.0, 1.0, 0.0),
      plan.Segment(plan.FALLBACK, 1.0, 2.0, 1.0),
  ]
  out = _assemble(segs, _const(1000, 1.0), _const(1000, 1.0), crossfade_s=0.1)
  fade = out[95:105]
  assert fade.max() > 1.2
  assert fade[0] == pytest.approx(1.0, abs=0.1)
  assert fade[-1] == pytest.approx(1.0, abs=0.1)


def test_crossfade_between_different_levels_is_monotonic():
  segs = [
      plan.Segment(plan.RIGHT, 0.0, 1.0, 0.0),
      plan.Segment(plan.FALLBACK, 1.0, 2.0, 1.0),
  ]
  out = _assemble(segs, _const(1000, 0.0), _const(1000, 1.0), crossfade_s=0.1)
  fade = out[95:105]
  assert np.all(np.diff(fade) > 0)
  assert 0.0 < fade[0] and fade[-1] < 1.0


def test_source_offsets_keep_samples_in_place_around_a_cut():
  segs = [
      plan.Segment(plan.RIGHT, 0.0, 1.0, 0.0),
      plan.Segment(plan.RIGHT, 1.0, 2.0, 5.0),
  ]
  out = _assemble(segs, _ramp(1000), _const(1000, 0.0), crossfade_s=0.1)
  np.testing.assert_array_equal(out[:95], np.arange(95))
  np.testing.assert_array_equal(out[105:], 500 + np.arange(105, 200) - 100)


def test_crossfade_shrinks_for_short_segments():
  segs = [
      plan.Segment(plan.RIGHT, 0.0, 1.0, 0.0),
      plan.Segment(plan.FALLBACK, 1.0, 1.04, 1.0),
      plan.Segment(plan.RIGHT, 1.04, 2.0, 1.04),
  ]
  out = _assemble(segs, _const(1000, 1.0), _const(1000, 2.0), crossfade_s=0.2)
  assert len(out) == 200
  assert np.isfinite(out).all()


def test_plan_not_starting_at_zero_is_placed_at_its_start():
  segs = [plan.Segment(plan.RIGHT, 3.0, 4.0, 1.0)]
  out = _assemble(segs, _ramp(1000), _const(1000, 0.0))
  np.testing.assert_array_equal(out, 100 + np.arange(100))


def test_source_range_outside_stream_is_silence():
  segs = [plan.Segment(plan.RIGHT, 0.0, 1.0, -0.5)]
  out = _assemble(segs, _const(1000, 1.0), _const(1000, 0.0))
  np.testing.assert_array_equal(out[:50], 0.0)
  np.testing.assert_array_equal(out[50:], 1.0)


def test_empty_plan_yields_nothing():
  assert not list(audio.AssembleAudio([], _const(1, 0), _const(1, 0), RATE))


def test_stream_source_matches_array_source():
  data = np.arange(1000, dtype=np.float32)[:, None] * [[1.0, -1.0]]
  chunks = iter(np.array_split(data, 37))
  stream = audio.StreamSource(chunks, channels=2, keep=50)
  array = audio.ArraySource(data)
  for start, end in [(-10, 20), (20, 120), (100, 400), (390, 900), (900, 1100)]:
    np.testing.assert_array_equal(stream.Read(start, end),
                                  array.Read(start, end))


def test_stream_source_cannot_go_back_beyond_keep():
  data = np.zeros((1000, 1), dtype=np.float32)
  stream = audio.StreamSource(iter(np.array_split(data, 10)), 1, keep=10)
  stream.Read(500, 600)
  stream.Read(495, 520)
  with pytest.raises(ValueError):
    stream.Read(100, 200)


def test_assembly_with_stream_sources_equals_array_sources():
  rng = np.random.default_rng(0)
  right = rng.standard_normal((1000, 2)).astype(np.float32)
  fallback = rng.standard_normal((1000, 2)).astype(np.float32)
  segs = [
      plan.Segment(plan.RIGHT, 0.0, 2.0, 0.0),
      plan.Segment(plan.FALLBACK, 2.0, 3.0, 2.0),
      plan.Segment(plan.RIGHT, 3.0, 5.0, 3.0),
      plan.Segment(plan.RIGHT, 5.0, 7.0, 6.0),
  ]
  expected = np.concatenate(
      list(
          audio.AssembleAudio(segs, audio.ArraySource(right),
                              audio.ArraySource(fallback), RATE)))
  got = np.concatenate(
      list(
          audio.AssembleAudio(
              segs,
              audio.StreamSource(iter(np.array_split(right, 13)), 2, RATE),
              audio.StreamSource(iter(np.array_split(fallback, 7)), 2, RATE),
              RATE,
              chunk_s=0.3)))
  np.testing.assert_allclose(got, expected)
