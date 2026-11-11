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
"""Shared fixtures. Files are written under the per-test ``tmp_path``, so the
suite is safe to run in parallel."""

import shutil
import subprocess

import pytest

FFMPEG = shutil.which('ffmpeg')


@pytest.fixture
def make_video(tmp_path):
  """Returns a factory that generates a short synthetic ``testsrc`` video
  with ffmpeg under this test's ``tmp_path`` and returns its path."""

  def _make(name, *, width=320, height=240, fps=24, duration=1.0):
    path = tmp_path / name
    subprocess.run([
        FFMPEG, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i',
        f'testsrc=size={width}x{height}:rate={fps}:duration={duration}', '-c:v',
        'libx264',
        str(path)
    ],
                   capture_output=True,
                   check=True,
                   text=True)
    return path

  return _make
