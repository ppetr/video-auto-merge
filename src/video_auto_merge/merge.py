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
"""Merges the audio of the right video into the left video.

The left video is kept unchanged and the chosen audio streams of the right
video are appended to it, re-timed onto the left timeline along the
alignment (see `plan` and `audio`).
"""
import logging
import pathlib
import tempfile
from typing import List, Optional

from video_auto_merge import align
from video_auto_merge import audio
from video_auto_merge import media
from video_auto_merge import plan
from video_auto_merge import report

_LOG = logging.getLogger(__name__)


def MergeAudio(
    left_video: pathlib.PurePath,
    right_video: pathlib.PurePath,
    alignment: align.VideoAlignment,
    output_path: pathlib.PurePath,
    right_streams: Optional[List[int]] = None,
    fallback_stream: Optional[int] = None,
    crossfade_s: float = audio.DEFAULT_CROSSFADE_S,
    min_match_s: float = plan.DEFAULT_MIN_MATCH_S,
    ffmpeg_path: pathlib.PurePath = pathlib.PurePath('ffmpeg'),
    ffprobe_path: pathlib.PurePath = pathlib.PurePath('ffprobe'),
) -> None:
  """Writes `left_video` with the audio of `right_video` attached.

  `right_streams` are file stream indices of the right video's audio (all
  of them by default); each becomes a new Opus track. Parts of the left
  timeline that the right video lacks are filled from the left video's
  audio stream `fallback_stream` (its first audio stream by default),
  converted to the layout of the right track.
  """
  blocks = report.OrderedBlocks(alignment.result)
  segments = plan.BuildPlan(blocks, alignment.left_times, alignment.right_times,
                            min_match_s)
  if not segments:
    raise ValueError('the alignment is empty')
  right_audio = media.ProbeAudioStreams(ffprobe_path, right_video)
  if right_streams is not None:
    right_audio = [s for s in right_audio if s.index in right_streams]
    if len(right_audio) != len(right_streams):
      raise ValueError(f'{right_video} has no audio streams {right_streams}')
  if not right_audio:
    raise ValueError(f'{right_video} has no audio streams to merge')
  if fallback_stream is None:
    left_audio = media.ProbeAudioStreams(ffprobe_path, left_video)
    if left_audio:
      fallback_stream = left_audio[0].index
  needs_fallback = any(s.source == plan.FALLBACK for s in segments)
  if needs_fallback and fallback_stream is None:
    raise ValueError(f'{left_video} has no audio stream for the gaps')

  with tempfile.TemporaryDirectory(prefix='video-auto-merge-') as work:
    tracks = []
    for stream in right_audio:
      _LOG.info('Assembling audio stream %d of %s (%d segments)', stream.index,
                right_video, len(segments))
      track = pathlib.Path(work) / f'audio{stream.index}.mka'
      right = audio.StreamSource(
          media.DecodePcm(ffmpeg_path, right_video, stream.index,
                          stream.sample_rate, stream.channels), stream.channels,
          stream.sample_rate)
      fallback = audio.StreamSource(
          media.DecodePcm(ffmpeg_path, left_video, fallback_stream,
                          stream.sample_rate, stream.channels)
          if needs_fallback else iter(()), stream.channels, stream.sample_rate)
      media.EncodeOpus(ffmpeg_path,
                       track,
                       audio.AssembleAudio(segments, right, fallback,
                                           stream.sample_rate, crossfade_s),
                       stream.sample_rate,
                       stream.channels,
                       language=stream.language,
                       title=stream.title)
      tracks.append(track)
    _LOG.info('Muxing %s', output_path)
    media.Mux(ffmpeg_path,
              left_video,
              tracks,
              output_path,
              offset_s=segments[0].start)
