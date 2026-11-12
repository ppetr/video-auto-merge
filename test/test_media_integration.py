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
"""Tests for media, running real ffmpeg/ffprobe."""

import shutil
import subprocess

import numpy as np
import pytest

from video_auto_merge import media

pytestmark = pytest.mark.skipif(not shutil.which('ffmpeg'),
                                reason='ffmpeg not on PATH')

RATE = 48000


def _ffmpeg(*args):
  subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', *args],
                 check=True)


@pytest.fixture(name='delayed_audio_video')
def _delayed_audio_video(tmp_path):
  """A 2 s video whose 1 s stereo 440 Hz audio starts 0.25 s late."""
  path = tmp_path / 'delayed.mkv'
  _ffmpeg('-f', 'lavfi', '-i', 'testsrc=size=160x120:rate=25:duration=2',
          '-itsoffset', '0.25', '-f', 'lavfi', '-i',
          f'sine=frequency=440:sample_rate={RATE}:duration=1', '-ac', '2',
          '-c:v', 'libx264', '-c:a', 'libopus', '-metadata:s:a:0',
          'language=ces', str(path))
  return path


def test_probe_audio_streams(delayed_audio_video):
  streams = media.ProbeAudioStreams('ffprobe', delayed_audio_video)
  assert len(streams) == 1
  assert streams[0].index == 1
  assert streams[0].channels == 2
  assert streams[0].sample_rate == RATE
  assert streams[0].language == 'ces'


def test_decode_pads_late_audio_start_with_silence(delayed_audio_video):
  pcm = np.concatenate(
      list(media.DecodePcm('ffmpeg', delayed_audio_video, 1, RATE, 2)))
  assert pcm.shape[1] == 2
  assert pcm.dtype == np.float32
  onset = np.nonzero(np.abs(pcm[:, 0]) > 0.02)[0][0]
  assert onset / RATE == pytest.approx(0.25, abs=0.005)
  assert len(pcm) == pytest.approx(1.25 * RATE, abs=1000)


def test_decode_downmixes_to_requested_channels(delayed_audio_video):
  pcm = np.concatenate(
      list(media.DecodePcm('ffmpeg', delayed_audio_video, 1, 24000, 1)))
  assert pcm.shape[1] == 1
  assert len(pcm) == pytest.approx(1.25 * 24000, abs=500)


def test_decode_reports_ffmpeg_errors(tmp_path):
  with pytest.raises(RuntimeError, match='decoding'):
    list(media.DecodePcm('ffmpeg', tmp_path / 'missing.mkv', 0, RATE, 2))


def test_encode_and_mux_adds_audio_stream(delayed_audio_video, tmp_path):
  tone = np.sin(2 * np.pi * 440 * np.arange(RATE) / RATE).astype(np.float32)
  chunks = np.array_split(np.stack([tone, tone], axis=1), 10)
  encoded = tmp_path / 'new.mka'
  media.EncodeOpus('ffmpeg', encoded, chunks, RATE, 2, language='slk')
  merged = tmp_path / 'merged.mkv'
  media.Mux('ffmpeg', delayed_audio_video, [encoded], merged)

  streams = media.ProbeAudioStreams('ffprobe', merged)
  assert [s.language for s in streams] == ['ces', 'slk']
  assert all(s.codec == 'opus' for s in streams)


@pytest.fixture(name='subtitled_video')
def _subtitled_video(tmp_path):
  srt = tmp_path / 'in.srt'
  srt.write_text('1\n00:00:00,500 --> 00:00:01,500\nHello\n\n')
  path = tmp_path / 'subtitled.mkv'
  _ffmpeg('-f', 'lavfi', '-i', 'testsrc=size=160x120:rate=25:duration=2', '-i',
          str(srt), '-c:v', 'libx264', '-c:s', 'srt', '-metadata:s:s:0',
          'language=eng', str(path))
  return path


def test_probe_subtitle_streams(subtitled_video):
  streams = media.ProbeSubtitleStreams('ffprobe', subtitled_video)
  assert len(streams) == 1
  assert streams[0].is_text
  assert streams[0].language == 'eng'


def test_extract_srt_returns_cues(subtitled_video):
  text = media.ExtractSrt('ffmpeg', subtitled_video, 1)
  assert '00:00:00,500 --> 00:00:01,500' in text
  assert 'Hello' in text


def test_wrapped_subtitle_track_is_muxed_with_metadata(subtitled_video,
                                                       tmp_path):
  srt = tmp_path / 'new.srt'
  srt.write_text('1\n00:00:00,100 --> 00:00:00,900\nAhoj\n\n')
  track = tmp_path / 'new.mks'
  media.WriteSubtitleTrack('ffmpeg', srt, track, language='ces')
  merged = tmp_path / 'merged.mkv'
  media.Mux('ffmpeg', subtitled_video, [track], merged)

  streams = media.ProbeSubtitleStreams('ffprobe', merged)
  assert [s.language for s in streams] == ['eng', 'ces']
