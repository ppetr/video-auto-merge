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
"""End-to-end tests running ffmpeg's real `signature` filter."""

import asyncio
import shutil

import pytest

from video_auto_merge import signatures

pytestmark = pytest.mark.skipif(not shutil.which('ffmpeg'),
                                reason='ffmpeg not on PATH')


def test_generate_and_load_signatures_roundtrip(make_video, tmp_path):
  # Regression test for the MediaTimeOfFrame/timebase quirk: .mp4 (not .mkv)
  # specifically exercises `GenerateSignature`'s forced 1 kHz timebase.
  clip = make_video('clip.mp4', width=160, height=120, fps=25, duration=4.0)
  xml_path = tmp_path / 'clip_sig.xml'

  asyncio.run(signatures.GenerateSignature('ffmpeg', clip, xml_path))
  times, sigs, confidences = signatures.LoadSignatures(xml_path,
                                                       reduce_seconds=1)

  assert sigs.shape == (4, signatures.SIGNATURE_LEN)
  assert confidences.shape == (4,)
  assert list(times) == [0.0, 1.0, 2.0, 3.0]


def test_load_signatures_returns_every_frame_by_default(make_video, tmp_path):
  clip = make_video('clip.mp4', width=160, height=120, fps=25, duration=2.0)
  xml_path = tmp_path / 'clip_sig.xml'

  asyncio.run(signatures.GenerateSignature('ffmpeg', clip, xml_path))
  times, sigs, _ = signatures.LoadSignatures(xml_path)

  assert sigs.shape == (50, signatures.SIGNATURE_LEN)
  assert times[:3].tolist() == pytest.approx([0.0, 0.04, 0.08])


def test_generate_signature_with_a_bracket_laden_filename(make_video, tmp_path):
  # Regression test: brackets/colons/commas in the output name broke
  # ffmpeg's filtergraph parser when embedded in `-vf` unescaped.
  clip = make_video('clip.mp4', width=160, height=120, fps=10, duration=2.0)
  xml_path = tmp_path / 'Film [FHD-1080p][HEVC-10bit]: cut, v2.xml'

  asyncio.run(signatures.GenerateSignature('ffmpeg', clip, xml_path))
  times, _, _ = signatures.LoadSignatures(xml_path, reduce_seconds=1)

  assert list(times) == [0.0, 1.0]
