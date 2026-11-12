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
"""End-to-end test of merging audio (real ffmpeg; the alignment is given)."""

import shutil
import subprocess

import numpy as np
import pytest

pytest.importorskip('jax')

# pylint: disable=wrong-import-position
from video_auto_merge import align
from video_auto_merge import frame_dp_align
from video_auto_merge import media
from video_auto_merge import merge
from video_auto_merge import subtitles

pytestmark = pytest.mark.skipif(not shutil.which('ffmpeg'),
                                reason='ffmpeg not on PATH')

RATE = 48000
FPS = 25


def _ffmpeg(*args):
  subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', *args],
                 check=True)


def _dominant_hz(pcm, start_s, end_s):
  window = pcm[round(start_s * RATE):round(end_s * RATE), 0]
  spectrum = np.abs(np.fft.rfft(window * np.hanning(len(window))))
  return np.argmax(spectrum) * RATE / len(window)


@pytest.fixture(name='videos')
def _videos(tmp_path):
  """Left: 4 s, stereo 220 Hz. Right: 6.0 s, mono, 880/1320/1760/2200 Hz in
  1.6/2.0/1.6/0.8 s parts."""
  left = tmp_path / 'left.mkv'
  _ffmpeg('-f', 'lavfi', '-i', f'testsrc=size=160x120:rate={FPS}:duration=4',
          '-f', 'lavfi', '-i', f'sine=frequency=220:sample_rate={RATE}', '-t',
          '4', '-ac', '2', '-c:v', 'libx264', '-c:a', 'libopus', str(left))
  right = tmp_path / 'right.mkv'
  srt = tmp_path / 'right.srt'
  srt.write_text(
      subtitles.FormatSrt([
          subtitles.Cue(0.5, 1.0, 'kept'),
          subtitles.Cue(2.0, 3.0, 'dropped'),
          subtitles.Cue(4.0, 4.5, 'moved'),
      ]))
  parts = [(880, 1.6), (1320, 2.0), (1760, 1.6), (2200, 0.8)]
  graph = ';'.join(f'sine=frequency={f}:sample_rate={RATE}:duration={d}[a{n}]'
                   for n, (f, d) in enumerate(parts))
  graph += ';' + ''.join(f'[a{n}]' for n in range(len(parts)))
  graph += f'concat=n={len(parts)}:v=0:a=1[out]'
  _ffmpeg('-f',
          'lavfi', '-i', f'testsrc=size=160x120:rate={FPS}:duration=6', '-i',
          str(srt), '-filter_complex', graph, '-map', '0:v', '-map', '[out]',
          '-map', '1', '-c:v', 'libx264', '-c:a', 'libopus', '-c:s', 'srt',
          '-metadata:s:a:0', 'language=slk', '-metadata:s:s:0', 'language=slk',
          str(right))
  return left, right


@pytest.fixture(name='alignment')
def _alignment():
  """Left frames 0-40 = right 0-40, left-only 40-60, right-only 40-90,
  left 60-100 = right 90-130, right-only 130-150."""
  result = frame_dp_align.FrameAlignmentResult(
      aligned=[
          frame_dp_align.AlignedBlock(0, 40, 0, 40),
          frame_dp_align.AlignedBlock(60, 100, 90, 130),
      ],
      unique=[
          frame_dp_align.UniqueBlock('left', 40, 60),
          frame_dp_align.UniqueBlock('right', 40, 90),
          frame_dp_align.UniqueBlock('right', 130, 150),
      ])
  return align.VideoAlignment(
      np.arange(100) / FPS,
      np.arange(150) / FPS, result)


def test_merge_places_right_audio_and_fills_the_gap(videos, alignment,
                                                    tmp_path):
  left, right = videos
  output = tmp_path / 'merged.mkv'

  merge.Merge(left, right, alignment, output)

  streams = media.ProbeAudioStreams('ffprobe', output)
  assert [s.language for s in streams] == [None, 'slk']
  assert streams[1].channels == 1
  pcm = np.concatenate(list(media.DecodePcm('ffmpeg', output, 2, RATE, 1)))
  assert len(pcm) == pytest.approx(4.0 * RATE, abs=0.1 * RATE)
  assert _dominant_hz(pcm, 0.2, 1.5) == pytest.approx(880, abs=10)
  assert _dominant_hz(pcm, 1.75, 2.25) == pytest.approx(220, abs=10)
  assert _dominant_hz(pcm, 2.6, 3.9) == pytest.approx(1760, abs=10)
  for start, end, hz in [(1.5, 1.59, 880), (1.61, 1.7, 220), (2.3, 2.39, 220),
                         (2.41, 2.5, 1760)]:
    assert _dominant_hz(pcm, start, end) == pytest.approx(hz, abs=40)


def test_merge_rejects_unknown_right_stream(videos, alignment, tmp_path):
  left, right = videos
  with pytest.raises(ValueError, match='no audio streams'):
    merge.Merge(left, right, alignment, tmp_path / 'out.mkv', right_audio=[7])


def test_merge_retimes_subtitles(videos, alignment, tmp_path):
  left, right = videos
  output = tmp_path / 'merged.mkv'

  merge.Merge(left, right, alignment, output)

  streams = media.ProbeSubtitleStreams('ffprobe', output)
  assert [s.language for s in streams] == ['slk']
  cues = subtitles.ParseSrt(media.ExtractSrt('ffmpeg', output,
                                             streams[0].index))
  assert [c.text for c in cues] == ['kept', 'moved']
  assert cues[0].start == pytest.approx(0.5, abs=0.01)
  assert cues[1].start == pytest.approx(2.8, abs=0.01)
  assert cues[1].end == pytest.approx(3.3, abs=0.01)


def test_merge_can_skip_subtitles(videos, alignment, tmp_path):
  left, right = videos
  output = tmp_path / 'merged.mkv'

  merge.Merge(left, right, alignment, output, right_subtitles=[])

  assert not media.ProbeSubtitleStreams('ffprobe', output)
