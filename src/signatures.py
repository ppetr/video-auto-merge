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
"""Per-frame video descriptors from ffmpeg's MPEG-7 `signature` filter.

`GenerateSignature` runs the filter over a video and writes its XML output;
`LoadSignatures` reads that XML back as numpy arrays (the input format of
`frame_dp_align`); `EnsureSignature` generates the XML only when a cached
copy (see `SignatureCachePath`) is missing or stale, since generation is
the slow, CPU-bound step of the whole pipeline.

The filter's own `detectmode`/`lookup` matching is intentionally not used -
only its raw per-frame `FrameSignature`/`FrameConfidence` data is read.

Requires `ffmpeg` to be installed and available in $PATH.
"""
import asyncio
import hashlib
import logging
import os
import pathlib
import re
import shlex
import subprocess
import tempfile
from typing import List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

import numpy as np
import numpy.typing as npt

SIGNATURE_LEN = 380

_SIGNATURE_NS = '{urn:mpeg:mpeg7:schema:2001}'


async def GenerateSignature(
    ffmpeg_path: pathlib.PurePath,
    input_path: pathlib.PurePath,
    output_xml_path: pathlib.PurePath,
    args: Union[str, List[str]] = (),
    pre_input_args: List[str] = (),
):
  """Runs ffmpeg's MPEG-7 `signature` filter over `input_path`, writing
  per-frame signatures to `output_xml_path` as XML. Decodes but otherwise
  discards the video itself (`-f null`).

  The filter's `MediaTimeOfFrame` is the frame's raw PTS *tick value*, not
  milliseconds - it only comes out right for Matroska input (whose
  container convention is a 1 kHz timebase) and is wrong for e.g. plain
  MP4. `settb=1/1000` forces the timebase before the frame reaches
  `signature`, so the times are correct for any container.

  The filter's own `filename=` option is parsed through ffmpeg's
  filtergraph mini-language, which a real release's filename routinely
  runs afoul of (`:`, `[`, `]`, `,`) - even quoting does not help. So the
  XML is written to a safe, randomly-named sibling file first and renamed
  to `output_xml_path` on success; a plain filesystem rename does not care
  what characters are in the name.

  Async so that the generation for two videos can run concurrently.
  """
  if isinstance(args, str):
    args = list(filter(None, args.split(' ')))
  output_xml_path = pathlib.Path(output_xml_path)
  output_xml_path.parent.mkdir(parents=True, exist_ok=True)
  fd, tmp_name = tempfile.mkstemp(dir=output_xml_path.parent, suffix='.xml')
  os.close(fd)
  tmp_path = pathlib.Path(tmp_name)
  try:
    cmd = [str(ffmpeg_path), '-loglevel', 'warning']
    cmd += pre_input_args
    cmd += ['-i', str(input_path)]
    cmd += args
    cmd += [
        '-vf', f'settb=1/1000,signature=format=xml:filename={tmp_path}', '-f',
        'null', '-'
    ]
    logging.info('Running command: %s', ' '.join(map(shlex.quote, cmd)))
    process = await asyncio.create_subprocess_exec(*cmd,
                                                   stdin=subprocess.DEVNULL)
    await process.wait()
    if process.returncode != 0:
      raise subprocess.CalledProcessError(process.returncode, cmd)
    os.replace(tmp_path, output_xml_path)
  finally:
    tmp_path.unlink(missing_ok=True)


def LoadSignatures(
    xml_path: pathlib.PurePath,
    reduce_seconds: Optional[float] = None,
) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
  """Streams a `GenerateSignature` XML file into numpy arrays.

  Returns `(times, signatures, confidences)`: `times` in seconds,
  `signatures` of shape `(N, SIGNATURE_LEN)` (`int16`, wide enough for safe
  L1-distance arithmetic), `confidences` of shape `(N,)`.

  By default every frame is returned. With `reduce_seconds`, only the first
  frame in each `reduce_seconds`-long bucket of video is kept.

  Streamed via `iterparse`+`clear()` - a feature film's XML can be ~200 MB
  and must never be loaded whole.
  """
  times = []
  sigs = []
  confidences = []
  last_bucket = None
  for _, elem in ET.iterparse(str(xml_path), events=('end',)):
    if elem.tag == _SIGNATURE_NS + 'VideoFrame':
      time_ms = int(elem.findtext(_SIGNATURE_NS + 'MediaTimeOfFrame'))
      bucket = (None if reduce_seconds is None else int(
          time_ms // (reduce_seconds * 1000)))
      if bucket is None or bucket != last_bucket:
        sig_text = elem.findtext(_SIGNATURE_NS + 'FrameSignature')
        sig = np.fromstring(sig_text, dtype=np.int16, sep=' ')
        if sig.size == SIGNATURE_LEN:
          confidence_text = elem.findtext(_SIGNATURE_NS + 'FrameConfidence')
          times.append(time_ms / 1000.0)
          sigs.append(sig)
          confidences.append(
              float(confidence_text) if confidence_text is not None else 1.0)
          last_bucket = bucket
      elem.clear()
  return (
      np.array(times, dtype=np.float64),
      np.array(sigs, dtype=np.int16) if sigs else np.zeros(
          (0, SIGNATURE_LEN), dtype=np.int16),
      np.array(confidences, dtype=np.float64),
  )


_UNSAFE_STEM_CHARS = re.compile(r'[^A-Za-z0-9._-]+')


def SignatureCachePath(cache_dir: pathlib.Path,
                       input_path: pathlib.PurePath) -> pathlib.Path:
  """Deterministic cache filename for `input_path`'s signature XML.

  Readable (a sanitized stem of `input_path` *as given*, e.g. a short
  symlink name rather than the messy name it resolves to) but
  collision-safe (a hash of the resolved absolute path, so a symlink and
  its target share one cache entry, while same-named files in different
  directories do not collide).
  """
  digest = hashlib.sha256(str(
      pathlib.Path(input_path).resolve()).encode()).hexdigest()[:16]
  safe_stem = _UNSAFE_STEM_CHARS.sub('_', pathlib.Path(input_path).stem)
  return cache_dir / f'{safe_stem}-{digest}.xml'


async def EnsureSignature(
    ffmpeg_path: pathlib.PurePath,
    input_path: pathlib.PurePath,
    xml_path: pathlib.Path,
    pre_input_args: List[str] = (),
) -> None:
  """Generates `xml_path` unless it already exists and is at least as new
  as `input_path` (Make-style staleness check)."""
  if xml_path.exists():
    if xml_path.stat().st_mtime >= pathlib.Path(input_path).stat().st_mtime:
      logging.info('Reusing cached signature %s', xml_path)
      return
    logging.info('%s is older than %s - regenerating', xml_path, input_path)
  await GenerateSignature(ffmpeg_path,
                          input_path,
                          xml_path,
                          pre_input_args=pre_input_args)
