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
"""Shared fixtures. Files are written under the per-test ``tmp_path``, so the
suite is safe to run in parallel."""

import dataclasses
import shutil
import subprocess
from typing import Tuple

import pytest

FFMPEG = shutil.which('ffmpeg')


@pytest.fixture
def make_video(tmp_path):
  """Returns a factory that generates a short synthetic ``testsrc`` video
  with ffmpeg under this test's ``tmp_path`` and returns its path."""

  def _make(name, *, width=320, height=240, fps=24, duration=1.0):
    path = tmp_path / name
    subprocess.run([
        FFMPEG, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i',
        f'testsrc=size={width}x{height}:rate={fps}:duration={duration}', '-c:v',
        'libx264',
        str(path)
    ],
                   capture_output=True,
                   check=True,
                   text=True)
    return path

  return _make


@dataclasses.dataclass
class InsertedSegmentVideos:
  """Two synthetic videos: `paths[1]` is `paths[0]` with an extra,
  visually distinct segment of `segment_seconds` inserted in the middle."""
  paths: Tuple
  fps: int
  segment_seconds: int


@pytest.fixture
def inserted_segment_videos(tmp_path):
  """Builds left = A + C and right = A + B + C from three distinct
  synthetic sources, so that B is present only in the right video."""
  fps = 25
  segment_seconds = 3

  def segment(source):
    path = tmp_path / f'{source}.mkv'
    subprocess.run([
        FFMPEG, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i',
        f'{source}=size=160x120:rate={fps}', '-t',
        str(segment_seconds), '-c:v', 'libx264', '-g', '1',
        str(path)
    ],
                   check=True)
    return path

  def concat(name, segments):
    listing = tmp_path / f'{name}.txt'
    listing.write_text(''.join(f"file '{s}'\n" for s in segments))
    path = tmp_path / f'{name}.mkv'
    subprocess.run([
        FFMPEG, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat',
        '-safe', '0', '-i',
        str(listing), '-c', 'copy',
        str(path)
    ],
                   check=True)
    return path

  first, extra, last = (
      segment(s) for s in ('testsrc', 'rgbtestsrc', 'mandelbrot'))
  return InsertedSegmentVideos(
      (concat('left', [first, last]), concat('right', [first, extra, last])),
      fps, segment_seconds)
