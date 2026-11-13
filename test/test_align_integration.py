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

import pytest

pytest.importorskip('jax')

from video_auto_merge import align  # noqa: E402  pylint: disable=wrong-import-position
from video_auto_merge import frame_dp_align  # noqa: E402  pylint: disable=wrong-import-position

pytestmark = pytest.mark.skipif(not shutil.which('ffmpeg'),
                                reason='ffmpeg not on PATH')


@pytest.mark.parametrize('max_base_cells,splits', [(None, False), (2000, True)])
def test_detects_a_segment_present_only_in_the_right_video(
    tmp_path, inserted_segment_videos, max_base_cells, splits):
  left, right = inserted_segment_videos.paths
  frame_dp_align.SPLIT_STATS.update(node=0, in_flight=0)

  alignment = align.AlignVideos(left,
                                right,
                                tmp_path / 'cache',
                                frame_dp_align.FrameDpParams(
                                    c0=500.0,
                                    c1=50.0,
                                    max_base_cells=max_base_cells),
                                pre_input_args=[])

  assert (sum(frame_dp_align.SPLIT_STATS.values()) > 0) == splits

  fps = inserted_segment_videos.fps
  seconds = inserted_segment_videos.segment_seconds
  assert len(alignment.left_times) == fps * 2 * seconds
  assert len(alignment.right_times) == fps * 3 * seconds
  unique = alignment.result.unique
  assert [u.side for u in unique] == ['right']
  assert unique[0].end - unique[0].start == pytest.approx(fps * seconds, abs=3)
  assert unique[0].start == pytest.approx(fps * seconds, abs=3)


def test_alignment_is_cached_per_parameters(tmp_path, monkeypatch,
                                            inserted_segment_videos):
  left, right = inserted_segment_videos.paths
  params = frame_dp_align.FrameDpParams(c0=500.0, c1=50.0)
  first = align.AlignVideos(left,
                            right,
                            tmp_path / 'cache',
                            params,
                            pre_input_args=[])

  def fail(*_):
    raise AssertionError('alignment was recomputed')

  monkeypatch.setattr(frame_dp_align, 'AlignFrames', fail)
  second = align.AlignVideos(left,
                             right,
                             tmp_path / 'cache',
                             params,
                             pre_input_args=[])

  assert second.result == first.result
  assert list(second.left_times) == list(first.left_times)
  assert list(second.right_times) == list(first.right_times)
  other = frame_dp_align.FrameDpParams(c0=501.0, c1=50.0)
  assert (align.AlignmentCachePath(tmp_path, left, right, other)
          != align.AlignmentCachePath(tmp_path, left, right, params))
