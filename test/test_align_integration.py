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
"""End-to-end test of `AlignVideos` with real ffmpeg and jax."""

import shutil
import subprocess

import pytest

pytest.importorskip('jax')

from video_auto_merge import align  # noqa: E402  pylint: disable=wrong-import-position
from video_auto_merge import frame_dp_align  # noqa: E402  pylint: disable=wrong-import-position

pytestmark = pytest.mark.skipif(not shutil.which('ffmpeg'),
                                reason='ffmpeg not on PATH')

_FPS = 25
_SEGMENT_SECONDS = 3


def _segment(tmp_path, source):
  path = tmp_path / f'{source}.mkv'
  subprocess.run([
      'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i',
      f'{source}=size=160x120:rate={_FPS}', '-t',
      str(_SEGMENT_SECONDS), '-c:v', 'libx264', '-g', '1',
      str(path)
  ],
                 check=True)
  return path


def _concat(tmp_path, name, segments):
  listing = tmp_path / f'{name}.txt'
  listing.write_text(''.join(f"file '{s}'\n" for s in segments))
  path = tmp_path / f'{name}.mkv'
  subprocess.run([
      'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat',
      '-safe', '0', '-i',
      str(listing), '-c', 'copy',
      str(path)
  ],
                 check=True)
  return path


def test_detects_a_segment_present_only_in_the_right_video(tmp_path):
  first = _segment(tmp_path, 'testsrc')
  extra = _segment(tmp_path, 'rgbtestsrc')
  last = _segment(tmp_path, 'mandelbrot')
  left = _concat(tmp_path, 'left', [first, last])
  right = _concat(tmp_path, 'right', [first, extra, last])

  alignment = align.AlignVideos(left,
                                right,
                                tmp_path / 'cache',
                                frame_dp_align.FrameDpParams(c0=500.0, c1=50.0),
                                pre_input_args=[])

  assert len(alignment.left_times) == _FPS * 2 * _SEGMENT_SECONDS
  assert len(alignment.right_times) == _FPS * 3 * _SEGMENT_SECONDS
  unique = alignment.result.unique
  assert [u.side for u in unique] == ['right']
  assert unique[0].end - unique[0].start == pytest.approx(_FPS *
                                                          _SEGMENT_SECONDS,
                                                          abs=3)
  assert unique[0].start == pytest.approx(_FPS * _SEGMENT_SECONDS, abs=3)
