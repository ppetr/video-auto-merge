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
"""Frame-level DP alignment between two per-frame descriptor sequences.

A frame-resolution alternative to a simpler, 1-second-block banded
Needleman-Wunsch-style alignment (`align_films.BandedAlign`, in the
sibling `avtools` project this was split out of): no band restriction -
every possible insertion/deletion is implicitly considered, not just
ones near a pre-estimated offset. See
`docs/frame_dp_align_notes.md` for why this was explored and what else was
tried; `docs/frame_dp_align_future_work.md` for what's deliberately not
included (constant-speed drift compensation in particular).

Cost model: between frame `i` of the left sequence and frame `j` of the
right sequence, the DP considers two kinds of edge:

  - match: `(i,j) -> (i+1,j+1)`, cost `D(i,j)` (the two frames'
    descriptor distance).
  - jump: `(i,j) -> (i+k,j)` or `(i,j) -> (i,j+k)` for any `k>0`, cost
    `c0 + c1*k` - content present in only one sequence (an insertion or
    removal), at a cost that is flat per jump plus linear in how many
    frames it skips, regardless of where it starts.

This module is built up in layers, each validated against the previous,
simpler one on small synthetic inputs before being trusted at scale:
a brute-force reference (`NaiveFrameDp`), an O(N*M) CPU version of the
same recurrence (`FastFrameDp`), linear-memory path reconstruction
(`Backtrack`), and a GPU/streaming version of all of the above
(`StreamingBacktrack`) usable on whole-film-length inputs.
"""
import numpy as np


def _CheckDpParams(c0, c1):
  if c0 < 0 or c1 < 0:
    raise ValueError(f'c0 and c1 must be non-negative (got c0={c0}, c1={c1})')


def NaiveFrameDp(D, c0, c1):
  """F(i,j) = min cost of a path from (0,0) to (i,j). O(N*M*(N+M)): every
  candidate jump origin is scanned explicitly, no shortcut - reference
  implementation for small inputs only, to validate `FastFrameDp`
  against."""
  _CheckDpParams(c0, c1)
  N, M = D.shape
  F = np.full((N + 1, M + 1), np.inf)
  F[0, 0] = 0.0
  for i in range(N + 1):
    for j in range(M + 1):
      if i == 0 and j == 0:
        continue
      best = np.inf
      if i > 0 and j > 0:
        best = min(best, F[i - 1, j - 1] + D[i - 1, j - 1])
      if i > 0:
        ip = np.arange(i)
        best = min(best, float((F[:i, j] + c0 + c1 * (i - ip)).min()))
      if j > 0:
        jp = np.arange(j)
        best = min(best, float((F[i, :j] + c0 + c1 * (j - jp)).min()))
      F[i, j] = best
  return F


def FastFrameDp(D, c0, c1):
  """Same recurrence as `NaiveFrameDp`, O(N*M): the "jump from anywhere
  before the current position" term collapses into a running prefix
  minimum rather than an explicit per-cell scan over every candidate
  origin.

  Derivation, for one row `i`: let `ColMinShifted(j) = min_{i'<i}
  [F(i',j) - c1*i']` (maintained incrementally across rows - the
  cross-row jump term for row `i` is then `c0 + c1*i + ColMinShifted(j)`,
  i.e. no re-scanning of earlier rows). Combine that with the match term
  into `A(j)`. The within-row jump term needs `min_{j'<j} F(i,j')`, but
  since a jump's cost is non-negative, `P(j) := min_{j'<=j} F(i,j')`
  satisfies `P(j) = min(A(j), P(j-1))` - i.e. `P` is simply the prefix
  minimum of `A` shifted by `c1*j` (same shift trick as the cross-row
  term), not a new dependency on `F` itself. So the whole row reduces to
  one forward pass over `A` (`np.minimum.accumulate`), not a sequential
  per-column loop - the same shape of computation
  `jax.lax.cummin`/`jax.lax.scan` run on a GPU (see `_HardRowStep`).
  """
  _CheckDpParams(c0, c1)
  N, M = D.shape
  j_idx = np.arange(M + 1, dtype=np.float64)

  F_prev = np.full(M + 1, np.inf)
  col_min_shifted = np.full(M + 1, np.inf)
  F = np.zeros((N + 1, M + 1))

  for i in range(N + 1):
    diag = np.full(M + 1, np.inf)
    diag[1:] = F_prev[:-1] + (D[i - 1, :] if i > 0 else 0.0)
    ijump = c0 + c1 * i + col_min_shifted
    A = np.minimum(diag, ijump)
    if i == 0:
      A[0] = 0.0  # true base case - no incoming edge at all.

    B = A - c1 * j_idx
    R = np.minimum.accumulate(B)
    R_shift = np.concatenate([[np.inf], R[:-1]])  # R(j-1), R(-1)=inf
    F_cur = np.minimum(A, c0 + c1 * j_idx + R_shift)

    F[i] = F_cur
    col_min_shifted = np.minimum(col_min_shifted, F_cur - c1 * i)
    F_prev = F_cur

  return F
