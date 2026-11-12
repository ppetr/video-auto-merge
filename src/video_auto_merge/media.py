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
"""Thin wrappers over ffprobe/ffmpeg: probing, PCM decoding, Opus encoding,
subtitle conversion and the final mux."""
import dataclasses
import json
import pathlib
import subprocess
import tempfile
from typing import Iterable, Iterator, List, Optional

import numpy as np

SAMPLE_BYTES = 4
DEFAULT_BITRATE_PER_CHANNEL = 64000
TEXT_SUBTITLE_CODECS = frozenset(
    ['subrip', 'ass', 'ssa', 'mov_text', 'webvtt', 'text'])


@dataclasses.dataclass
class AudioStreamInfo:
  """An audio stream of a file: `index` is the stream's index in the file
  (as used by `-map 0:<index>`)."""
  index: int
  codec: str
  channels: int
  sample_rate: int
  language: Optional[str] = None
  title: Optional[str] = None


@dataclasses.dataclass
class SubtitleStreamInfo:
  """A subtitle stream of a file. Only text codecs can be re-timed."""
  index: int
  codec: str
  language: Optional[str] = None
  title: Optional[str] = None

  @property
  def is_text(self) -> bool:
    return self.codec in TEXT_SUBTITLE_CODECS


def _ProbeStreams(ffprobe_path: pathlib.PurePath, path: pathlib.PurePath,
                  selector: str) -> List[dict]:
  out = subprocess.run([
      str(ffprobe_path), '-v', 'error', '-print_format', 'json',
      '-show_streams', '-select_streams', selector,
      str(path)
  ],
                       check=True,
                       capture_output=True).stdout
  return json.loads(out)['streams']


def ProbeAudioStreams(ffprobe_path: pathlib.PurePath,
                      path: pathlib.PurePath) -> List[AudioStreamInfo]:
  """Lists the audio streams of `path`."""
  return [
      AudioStreamInfo(index=s['index'],
                      codec=s['codec_name'],
                      channels=s['channels'],
                      sample_rate=int(s['sample_rate']),
                      language=s.get('tags', {}).get('language'),
                      title=s.get('tags', {}).get('title'))
      for s in _ProbeStreams(ffprobe_path, path, 'a')
  ]


def ProbeSubtitleStreams(ffprobe_path: pathlib.PurePath,
                         path: pathlib.PurePath) -> List[SubtitleStreamInfo]:
  """Lists the subtitle streams of `path`."""
  return [
      SubtitleStreamInfo(index=s['index'],
                         codec=s.get('codec_name', ''),
                         language=s.get('tags', {}).get('language'),
                         title=s.get('tags', {}).get('title'))
      for s in _ProbeStreams(ffprobe_path, path, 's')
  ]


def _Check(returncode: int, stderr, what: str) -> None:
  if returncode:
    stderr.seek(0)
    message = stderr.read().decode(errors='replace')
    raise RuntimeError(f'{what} failed ({returncode}): {message}')


def DecodePcm(ffmpeg_path: pathlib.PurePath,
              path: pathlib.PurePath,
              stream_index: int,
              rate: int,
              channels: int,
              chunk_samples: int = 48000) -> Iterator[np.ndarray]:
  """Decodes one audio stream to float32 `(samples, channels)` chunks.

  Sample 0 is the container's time 0: a stream that starts later is padded
  with silence at the start (and any samples before time 0 are trimmed).
  """
  cmd = [
      str(ffmpeg_path), '-v', 'error', '-nostdin', '-i',
      str(path), '-map', f'0:{stream_index}', '-vn', '-sn', '-af',
      'aresample=async=1:first_pts=0', '-ar',
      str(rate), '-ac',
      str(channels), '-f', 'f32le', 'pipe:1'
  ]
  frame_bytes = channels * SAMPLE_BYTES
  with tempfile.TemporaryFile() as stderr:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=stderr)
    try:
      while True:
        data = proc.stdout.read(chunk_samples * frame_bytes)
        if not data:
          break
        usable = len(data) - len(data) % frame_bytes
        yield np.frombuffer(data[:usable],
                            dtype=np.float32).reshape(-1, channels)
      proc.wait()
      _Check(proc.returncode, stderr, 'ffmpeg decoding')
    finally:
      if proc.poll() is None:
        proc.kill()
        proc.wait()
      proc.stdout.close()


def EncodeOpus(ffmpeg_path: pathlib.PurePath,
               output_path: pathlib.PurePath,
               chunks: Iterable[np.ndarray],
               rate: int,
               channels: int,
               bitrate: Optional[int] = None,
               language: Optional[str] = None,
               title: Optional[str] = None) -> None:
  """Encodes float32 `(samples, channels)` chunks to a VBR Opus file; the
  container is chosen by the extension of `output_path` (e.g. `.mka`)."""
  if bitrate is None:
    bitrate = DEFAULT_BITRATE_PER_CHANNEL * channels
  cmd = [
      str(ffmpeg_path), '-v', 'error', '-nostdin', '-y', '-f', 'f32le', '-ar',
      str(rate), '-ac',
      str(channels), '-i', 'pipe:0', '-c:a', 'libopus', '-vbr', 'on', '-b:a',
      str(bitrate)
  ]
  for key, value in (('language', language), ('title', title)):
    if value:
      cmd += ['-metadata:s:a:0', f'{key}={value}']
  cmd.append(str(output_path))
  with tempfile.TemporaryFile() as stderr:
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=stderr)
    try:
      for chunk in chunks:
        proc.stdin.write(
            np.ascontiguousarray(chunk, dtype=np.float32).tobytes())
      proc.stdin.close()
      proc.wait()
      _Check(proc.returncode, stderr, 'ffmpeg encoding')
    finally:
      if proc.poll() is None:
        proc.kill()
        proc.wait()


def ExtractSrt(ffmpeg_path: pathlib.PurePath, path: pathlib.PurePath,
               stream_index: int) -> str:
  """Returns a text subtitle stream of `path` converted to SRT."""
  with tempfile.TemporaryFile() as stderr:
    proc = subprocess.run([
        str(ffmpeg_path), '-v', 'error', '-nostdin', '-i',
        str(path), '-map', f'0:{stream_index}', '-c:s', 'srt', '-f', 'srt',
        'pipe:1'
    ],
                          stdout=subprocess.PIPE,
                          stderr=stderr,
                          check=False)
    _Check(proc.returncode, stderr, 'ffmpeg subtitle extraction')
  return proc.stdout.decode('utf-8', errors='replace')


def WriteSubtitleTrack(ffmpeg_path: pathlib.PurePath,
                       srt_path: pathlib.PurePath,
                       output_path: pathlib.PurePath,
                       language: Optional[str] = None,
                       title: Optional[str] = None) -> None:
  """Wraps an SRT file into a Matroska subtitle file with metadata."""
  cmd = [
      str(ffmpeg_path), '-v', 'error', '-nostdin', '-y', '-i',
      str(srt_path), '-c', 'copy', '-f', 'matroska'
  ]
  for key, value in (('language', language), ('title', title)):
    if value:
      cmd += ['-metadata:s:s:0', f'{key}={value}']
  cmd.append(str(output_path))
  with tempfile.TemporaryFile() as stderr:
    proc = subprocess.run(cmd, stderr=stderr, check=False)
    _Check(proc.returncode, stderr, 'ffmpeg subtitle wrapping')


def Mux(ffmpeg_path: pathlib.PurePath,
        base: pathlib.PurePath,
        extra_inputs: List[pathlib.PurePath],
        output_path: pathlib.PurePath,
        offset_s: float = 0.0) -> None:
  """Copies every stream of `base` and appends all streams of each file in
  `extra_inputs` (delayed by `offset_s`), without re-encoding."""
  cmd = [str(ffmpeg_path), '-v', 'error', '-nostdin', '-y', '-i', str(base)]
  for path in extra_inputs:
    cmd += ['-itsoffset', f'{offset_s:.6f}', '-i', str(path)]
  cmd += ['-map', '0']
  for n in range(len(extra_inputs)):
    cmd += ['-map', str(n + 1)]
  cmd += ['-c', 'copy', str(output_path)]
  with tempfile.TemporaryFile() as stderr:
    proc = subprocess.run(cmd, stderr=stderr, check=False)
    _Check(proc.returncode, stderr, 'ffmpeg mux')
