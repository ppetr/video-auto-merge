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
"""Merges streams of the right video into the left video.

The left video is kept unchanged. Audio and text subtitle streams of the
right video are appended to it, re-timed onto the left timeline along the
alignment (see `plan`, `audio` and `subtitles`).
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
from video_auto_merge import subtitles

_LOG = logging.getLogger(__name__)


def _SelectAudio(ffprobe_path: pathlib.PurePath, video: pathlib.PurePath,
                 wanted: Optional[List[int]]) -> List[media.AudioStreamInfo]:
  streams = media.ProbeAudioStreams(ffprobe_path, video)
  if wanted is not None:
    streams = [s for s in streams if s.index in wanted]
    if len(streams) != len(wanted):
      raise ValueError(f'{video} has no audio streams {wanted}')
  if not streams:
    raise ValueError(f'{video} has no audio streams to merge')
  return streams


def _SelectSubtitles(
    ffprobe_path: pathlib.PurePath, video: pathlib.PurePath,
    wanted: Optional[List[int]]) -> List[media.SubtitleStreamInfo]:
  streams = media.ProbeSubtitleStreams(ffprobe_path, video)
  if wanted is not None:
    streams = [s for s in streams if s.index in wanted]
    if len(streams) != len(wanted):
      raise ValueError(f'{video} has no subtitle streams {wanted}')
    image = [s.index for s in streams if not s.is_text]
    if image:
      raise ValueError(f'subtitle streams {image} of {video} are not text')
    return streams
  for stream in streams:
    if not stream.is_text:
      _LOG.warning('Skipping image subtitle stream %d (%s) of %s', stream.index,
                   stream.codec, video)
  return [s for s in streams if s.is_text]


def _AudioTrack(work: pathlib.Path, segments: List[plan.Segment],
                stream: media.AudioStreamInfo, right_video: pathlib.PurePath,
                left_video: pathlib.PurePath, fallback_stream: Optional[int],
                crossfade_s: float,
                ffmpeg_path: pathlib.PurePath) -> pathlib.Path:
  _LOG.info('Assembling audio stream %d of %s (%d segments)', stream.index,
            right_video, len(segments))
  rate, channels = stream.sample_rate, stream.channels
  right = audio.StreamSource(
      media.DecodePcm(ffmpeg_path, right_video, stream.index, rate, channels),
      channels, rate)
  if fallback_stream is None:
    fallback_chunks = iter(())
  else:
    fallback_chunks = media.DecodePcm(ffmpeg_path, left_video, fallback_stream,
                                      rate, channels)
  fallback = audio.StreamSource(fallback_chunks, channels, rate)
  track = work / f'audio{stream.index}.mka'
  media.EncodeOpus(ffmpeg_path,
                   track,
                   audio.AssembleAudio(segments, right, fallback, rate,
                                       crossfade_s),
                   rate,
                   channels,
                   language=stream.language,
                   title=stream.title)
  return track


def _SubtitleTrack(work: pathlib.Path, segments: List[plan.Segment],
                   stream: media.SubtitleStreamInfo,
                   right_video: pathlib.PurePath,
                   ffmpeg_path: pathlib.PurePath) -> pathlib.Path:
  _LOG.info('Re-timing subtitle stream %d of %s', stream.index, right_video)
  cues = subtitles.RemapCues(
      subtitles.ParseSrt(
          media.ExtractSrt(ffmpeg_path, right_video, stream.index)), segments)
  # The mux delays every extra input by the plan's start, so cues are
  # written relative to it.
  for cue in cues:
    cue.start -= segments[0].start
    cue.end -= segments[0].start
  srt = work / f'subtitles{stream.index}.srt'
  srt.write_text(subtitles.FormatSrt(cues), encoding='utf-8')
  track = srt.with_suffix('.mks')
  media.WriteSubtitleTrack(ffmpeg_path,
                           srt,
                           track,
                           language=stream.language,
                           title=stream.title)
  return track


def Merge(
    left_video: pathlib.PurePath,
    right_video: pathlib.PurePath,
    alignment: align.VideoAlignment,
    output_path: pathlib.PurePath,
    right_audio: Optional[List[int]] = None,
    right_subtitles: Optional[List[int]] = None,
    fallback_stream: Optional[int] = None,
    crossfade_s: float = audio.DEFAULT_CROSSFADE_S,
    min_match_s: float = plan.DEFAULT_MIN_MATCH_S,
    ffmpeg_path: pathlib.PurePath = pathlib.PurePath('ffmpeg'),
    ffprobe_path: pathlib.PurePath = pathlib.PurePath('ffprobe'),
) -> None:
  """Writes `left_video` with streams of `right_video` attached.

  `right_audio` and `right_subtitles` are file stream indices in the right
  video (by default all audio streams and all text subtitle streams; `[]`
  selects none of the subtitles). Each audio stream becomes a new Opus
  track. Parts of the left timeline that the right video lacks are filled
  from the left video's audio stream `fallback_stream` (its first audio
  stream by default), converted to the layout of the right track.
  """
  blocks = report.OrderedBlocks(alignment.result)
  segments = plan.BuildPlan(blocks, alignment.left_times, alignment.right_times,
                            min_match_s)
  if not segments:
    raise ValueError('the alignment is empty')
  audio_streams = _SelectAudio(ffprobe_path, right_video, right_audio)
  subtitle_streams = _SelectSubtitles(ffprobe_path, right_video,
                                      right_subtitles)
  needs_fallback = any(s.source == plan.FALLBACK for s in segments)
  if needs_fallback and fallback_stream is None:
    left_audio = media.ProbeAudioStreams(ffprobe_path, left_video)
    if not left_audio:
      raise ValueError(f'{left_video} has no audio stream for the gaps')
    fallback_stream = left_audio[0].index
  if not needs_fallback:
    fallback_stream = None

  with tempfile.TemporaryDirectory(prefix='video-auto-merge-') as temp:
    work = pathlib.Path(temp)
    tracks = [
        _AudioTrack(work, segments, stream, right_video, left_video,
                    fallback_stream, crossfade_s, ffmpeg_path)
        for stream in audio_streams
    ]
    tracks += [
        _SubtitleTrack(work, segments, stream, right_video, ffmpeg_path)
        for stream in subtitle_streams
    ]
    _LOG.info('Muxing %s', output_path)
    media.Mux(ffmpeg_path,
              left_video,
              tracks,
              output_path,
              offset_s=segments[0].start)
