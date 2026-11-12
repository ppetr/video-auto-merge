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
"""End-to-end test of the command-line interface (real ffmpeg and jax)."""

import shutil
import subprocess

import pytest

pytest.importorskip('jax')

from video_auto_merge import cli  # noqa: E402  pylint: disable=wrong-import-position

pytestmark = pytest.mark.skipif(not shutil.which('ffmpeg'),
                                reason='ffmpeg not on PATH')


def _run(tmp_path, videos, *extra):
  left, right = videos.paths
  return cli.main([
      'align',
      str(left),
      str(right), '--cache-dir',
      str(tmp_path / 'cache'), '--hwaccel', 'none', '--c0', '500', '--c1', '50',
      *extra
  ])


def test_align_prints_only_the_differences_by_default(tmp_path, capsys,
                                                      inserted_segment_videos):
  assert _run(tmp_path, inserted_segment_videos) == 0

  lines = capsys.readouterr().out.splitlines()
  assert lines[0].split()[:3] == ['#', 'kind', 'out']
  assert len(lines) == 2
  assert lines[1].split()[1] == 'right'


def test_align_all_lists_matched_blocks_too(tmp_path, capsys,
                                            inserted_segment_videos):
  assert _run(tmp_path, inserted_segment_videos, '--all') == 0

  kinds = [l.split()[1] for l in capsys.readouterr().out.splitlines()[1:]]
  assert kinds == ['both', 'right', 'both']


def test_align_writes_the_table_to_a_file(tmp_path, capsys,
                                          inserted_segment_videos):
  output = tmp_path / 'table.txt'

  assert _run(tmp_path, inserted_segment_videos, '-o', str(output)) == 0

  assert capsys.readouterr().out == ''
  assert 'right' in output.read_text()


def _with_audio(video, frequency):
  out = video.with_name(f'{video.stem}-audio.mkv')
  subprocess.run([
      'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i',
      str(video), '-f', 'lavfi', '-i', f'sine=frequency={frequency}',
      '-shortest', '-c:v', 'copy', '-c:a', 'libopus',
      str(out)
  ],
                 check=True)
  return out


def test_merge_attaches_right_audio_to_left_video(tmp_path,
                                                  inserted_segment_videos):
  left, right = inserted_segment_videos.paths
  left, right = _with_audio(left, 220), _with_audio(right, 880)
  output = tmp_path / 'merged.mkv'

  assert cli.main([
      'merge',
      str(left),
      str(right),
      str(output), '--cache-dir',
      str(tmp_path / 'cache'), '--hwaccel', 'none', '--c0', '500', '--c1', '50'
  ]) == 0

  probe = subprocess.run([
      'ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type', '-of',
      'csv=p=0',
      str(output)
  ],
                         check=True,
                         capture_output=True,
                         text=True).stdout.split()
  assert probe == ['video', 'audio', 'audio']
