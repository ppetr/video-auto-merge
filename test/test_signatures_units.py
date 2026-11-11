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
"""Unit tests for signatures' pure logic (no ffmpeg involved)."""

import asyncio
import os

import numpy as np

import signatures


class TestLoadSignatures:
  """Tests for `LoadSignatures` on small hand-written XML."""

  def _write_xml(self, path, times_ms):
    sig = ' '.join(['1'] * signatures.SIGNATURE_LEN)
    frames = ''.join(f'<VideoFrame><MediaTimeOfFrame>{t}</MediaTimeOfFrame>'
                     f'<FrameSignature>{sig}</FrameSignature>'
                     '<FrameConfidence>2</FrameConfidence></VideoFrame>'
                     for t in times_ms)
    path.write_text('<Mpeg7 xmlns="urn:mpeg:mpeg7:schema:2001">' + frames +
                    '</Mpeg7>')

  def test_keeps_every_frame_by_default(self, tmp_path):
    xml_path = tmp_path / 'sig.xml'
    self._write_xml(xml_path, [0, 40, 80, 1000])

    times, sigs, confidences = signatures.LoadSignatures(xml_path)

    assert list(times) == [0.0, 0.04, 0.08, 1.0]
    assert sigs.shape == (4, signatures.SIGNATURE_LEN)
    assert sigs.dtype == np.int16
    assert list(confidences) == [2.0] * 4

  def test_reduce_seconds_keeps_first_frame_per_bucket(self, tmp_path):
    xml_path = tmp_path / 'sig.xml'
    self._write_xml(xml_path, [0, 40, 80, 1000, 1040, 2500])

    times, _, _ = signatures.LoadSignatures(xml_path, reduce_seconds=1)

    assert list(times) == [0.0, 1.0, 2.5]

  def test_empty_file(self, tmp_path):
    xml_path = tmp_path / 'sig.xml'
    self._write_xml(xml_path, [])

    times, sigs, confidences = signatures.LoadSignatures(xml_path)

    assert times.shape == (0,)
    assert sigs.shape == (0, signatures.SIGNATURE_LEN)
    assert confidences.shape == (0,)


class TestSignatureCachePath:
  """Tests for `SignatureCachePath`."""

  def test_deterministic(self, tmp_path):
    cache_dir = tmp_path / 'cache'
    a = signatures.SignatureCachePath(cache_dir, tmp_path / 'film.mkv')
    b = signatures.SignatureCachePath(cache_dir, tmp_path / 'film.mkv')
    assert a == b

  def test_same_basename_different_directory_does_not_collide(self, tmp_path):
    cache_dir = tmp_path / 'cache'
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    a = signatures.SignatureCachePath(cache_dir, tmp_path / 'a' / 'film.mkv')
    b = signatures.SignatureCachePath(cache_dir, tmp_path / 'b' / 'film.mkv')
    assert a != b

  def test_uses_the_given_name_not_a_symlink_target(self, tmp_path):
    cache_dir = tmp_path / 'cache'
    target = tmp_path / 'Film [FHD-1080p][HEVC].mkv'
    target.write_text('source')
    link = tmp_path / 'film1.mkv'
    link.symlink_to(target)

    path = signatures.SignatureCachePath(cache_dir, link)

    assert path.name.startswith('film1-')
    assert '[' not in path.name and ']' not in path.name

  def test_a_symlink_and_its_target_share_one_cache_entry(self, tmp_path):
    cache_dir = tmp_path / 'cache'
    target = tmp_path / 'film.mkv'
    target.write_text('source')
    link = tmp_path / 'alias.mkv'
    link.symlink_to(target)

    via_target = signatures.SignatureCachePath(cache_dir, target)
    via_link = signatures.SignatureCachePath(cache_dir, link)

    assert via_target.name.rsplit('-', 1)[1] == via_link.name.rsplit('-', 1)[1]


class TestEnsureSignature:
  """Tests for `EnsureSignature` with `GenerateSignature` faked."""

  def _patch_generate(self, monkeypatch):
    calls = []

    async def fake_generate(ffmpeg_path,
                            input_path,
                            xml_path,
                            pre_input_args=()):
      # pylint: disable=unused-argument
      calls.append((input_path, xml_path))
      xml_path.write_text('<fake/>')

    monkeypatch.setattr(signatures, 'GenerateSignature', fake_generate)
    return calls

  def test_generates_when_no_cache_exists(self, tmp_path, monkeypatch):
    calls = self._patch_generate(monkeypatch)
    src = tmp_path / 'film.mkv'
    src.write_text('source')
    xml_path = tmp_path / 'cached.xml'

    asyncio.run(signatures.EnsureSignature('ffmpeg', src, xml_path))

    assert len(calls) == 1
    assert xml_path.exists()

  def test_reuses_a_fresh_cache_without_regenerating(self, tmp_path,
                                                     monkeypatch):
    calls = self._patch_generate(monkeypatch)
    src = tmp_path / 'film.mkv'
    src.write_text('source')
    xml_path = tmp_path / 'cached.xml'

    asyncio.run(signatures.EnsureSignature('ffmpeg', src, xml_path))
    asyncio.run(signatures.EnsureSignature('ffmpeg', src, xml_path))

    assert len(calls) == 1

  def test_regenerates_when_source_is_newer_than_the_cache(
      self, tmp_path, monkeypatch):
    calls = self._patch_generate(monkeypatch)
    src = tmp_path / 'film.mkv'
    src.write_text('source')
    xml_path = tmp_path / 'cached.xml'

    asyncio.run(signatures.EnsureSignature('ffmpeg', src, xml_path))
    old = xml_path.stat().st_mtime - 10
    os.utime(xml_path, (old, old))

    asyncio.run(signatures.EnsureSignature('ffmpeg', src, xml_path))

    assert len(calls) == 2
