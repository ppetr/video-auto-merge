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
"""Re-runs test_frame_dp_align_units.py's entire suite gated on jax
actually having a real GPU device selected.

Excluded from the default `pytest`/`pytest test/` run (see the `gpu`
marker and `addopts` in pyproject.toml) - a routine test run should not
require a GPU, and silently running "GPU tests" on a CPU-only jax
install would defeat the point. Run explicitly:

    pytest -m gpu -v

(plain `pytest test/test_frame_dp_align_gpu.py` without `-m gpu` will
report 0 selected - `addopts` deselects `gpu`-marked tests regardless
of which file they came from, so the marker must be given explicitly.)

This file adds no new test logic - every test below is the *same*
function object imported from test_frame_dp_align_units.py, just
collected again here under the `gpu` marker (via the module-level
`pytestmark`) and gated by `require_gpu`, an autouse fixture that fails
loudly if jax resolves to a CPU-only device instead of silently running
on CPU under a false "GPU-tested" label. See README.md's "Testing"
section for how to install a CUDA-enabled jaxlib.
"""
import pytest

jax = pytest.importorskip('jax')

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def require_gpu():
  gpu_devices = [d for d in jax.devices() if d.platform != 'cpu']
  if not gpu_devices:
    pytest.fail(
        f'No GPU device available to jax (jax.devices()={jax.devices()}) '
        "- install a CUDA-enabled jaxlib (see README.md's Testing "
        'section) before running the gpu-marked suite.')


# pylint: disable=unused-import,wrong-import-position
from test_frame_dp_align_units import (
    test_fast_matches_naive,
    test_base_case,
    test_rejects_negative_params,
    test_identical_sequences_prefer_matching,
    test_backtrack_matches_naive_optimum,
    test_backtrack_exercises_in_flight_split,
    test_streaming_backtrack_matches_naive_optimum,
    test_align_frames_finds_a_right_side_insert,
    test_one_hot_cost_transform_gemm_matches_l1,
)
