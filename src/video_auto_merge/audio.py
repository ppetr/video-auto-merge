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
"""Assembles an audio stream along a segment plan, with crossfades at joins.

`AssembleAudio` yields the output as float32 chunks of shape
`(samples, channels)`. Each join between two segments of the plan is a
symmetric crossfade centred on the join (equal-power quarter-sine
curves), so the output has exactly the length of the plan and every
sample stays at its position on the left timeline.
"""
import dataclasses
from typing import Iterator, List

import numpy as np

from video_auto_merge import plan

DEFAULT_CROSSFADE_S = 0.02
DEFAULT_CHUNK_S = 1.0


class SampleSource:
  """Random-access-looking reader of a decoded audio stream. `Read` returns
  float32 `(end - start, channels)`; samples outside the stream are zero."""

  def Read(self, start: int, end: int) -> np.ndarray:
    raise NotImplementedError


class ArraySource(SampleSource):
  """A source backed by an in-memory `(samples, channels)` array."""

  def __init__(self, samples: np.ndarray):
    self._samples = np.asarray(samples, dtype=np.float32)

  def Read(self, start: int, end: int) -> np.ndarray:
    out = np.zeros((end - start, self._samples.shape[1]), dtype=np.float32)
    lo = max(start, 0)
    hi = min(end, len(self._samples))
    if hi > lo:
      out[lo - start:hi - start] = self._samples[lo:hi]
    return out


class StreamSource(SampleSource):
  """A source fed by an iterator of consecutive `(samples, channels)` chunks
  (e.g. a decoder pipe). Reads must move forward; going back is possible
  only by `keep` samples before the furthest start requested so far."""

  def __init__(self, chunks: Iterator[np.ndarray], channels: int, keep: int):
    self._chunks = chunks
    self._channels = channels
    self._keep = keep
    self._buffer = np.zeros((0, channels), dtype=np.float32)
    self._start = 0
    self._eof = False

  @property
  def _end(self) -> int:
    return self._start + len(self._buffer)

  def _Trim(self, keep_from: int) -> None:
    drop = min(keep_from - self._start, len(self._buffer))
    if drop > 0:
      self._buffer = self._buffer[drop:]
      self._start += drop

  def Read(self, start: int, end: int) -> np.ndarray:
    out = np.zeros((end - start, self._channels), dtype=np.float32)
    lo = max(start, 0)
    if end <= lo:
      return out
    if lo < self._start:
      raise ValueError(f'cannot read back to {lo}, buffer starts at '
                       f'{self._start}')
    while self._end < end and not self._eof:
      try:
        chunk = next(self._chunks)
      except StopIteration:
        self._eof = True
        break
      self._buffer = np.concatenate([self._buffer, chunk])
      self._Trim(lo - self._keep)
    self._Trim(lo - self._keep)
    hi = min(end, self._end)
    if hi > lo:
      out[lo - start:hi - start] = self._buffer[lo - self._start:hi -
                                                self._start]
    return out


@dataclasses.dataclass
class _Placed:
  """A plan segment in samples: dest `[start, end)` and the offset to add to
  a dest sample to get the source sample."""
  source: SampleSource
  start: int
  end: int
  offset: int


def _Place(segments: List[plan.Segment], right: SampleSource,
           fallback: SampleSource, rate: int) -> List[_Placed]:
  origin = segments[0].start
  placed = []
  for seg in segments:
    start = round((seg.start - origin) * rate)
    end = round((seg.end - origin) * rate)
    placed.append(
        _Placed(right if seg.source == plan.RIGHT else fallback, start, end,
                round(seg.source_start * rate) - start))
  return placed


def _FadeCurves(length: int):
  x = (np.arange(length, dtype=np.float32) + 0.5) / length
  return (np.sin(0.5 * np.pi * x)[:, None], np.cos(0.5 * np.pi * x)[:, None])


def AssembleAudio(segments: List[plan.Segment],
                  right: SampleSource,
                  fallback: SampleSource,
                  rate: int,
                  crossfade_s: float = DEFAULT_CROSSFADE_S,
                  chunk_s: float = DEFAULT_CHUNK_S) -> Iterator[np.ndarray]:
  """Yields the assembled audio, in order, as float32 `(samples, channels)`.

  Segment contents are read from `right` or `fallback` at `rate` samples per
  second. A crossfade is at most `crossfade_s` long and is shortened where a
  neighbouring segment is too short to hold it. The output has exactly
  `round((segments[-1].end - segments[0].start) * rate)` samples.
  """
  if not segments:
    return
  placed = _Place(segments, right, fallback, rate)
  chunk = max(1, round(chunk_s * rate))
  half = [
      min(round(crossfade_s * rate / 2),
          min(a.end - a.start, b.end - b.start) // 2)
      for a, b in zip(placed, placed[1:])
  ]
  for k, seg in enumerate(placed):
    head = half[k - 1] if k > 0 else 0
    tail = half[k] if k < len(half) else 0
    if head:
      fade_in, fade_out = _FadeCurves(2 * head)
      lo = seg.start - head
      prev = placed[k - 1]
      mixed = (prev.source.Read(lo + prev.offset, lo + 2 * head + prev.offset) *
               fade_out +
               seg.source.Read(lo + seg.offset, lo + 2 * head + seg.offset) *
               fade_in)
      yield mixed
    pos = seg.start + head
    while pos < seg.end - tail:
      stop = min(pos + chunk, seg.end - tail)
      data = seg.source.Read(pos + seg.offset, stop + seg.offset)
      yield data
      pos = stop
