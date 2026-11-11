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

# Diagnostic counters (not thread-safe - fine for single-process tests/
# scripts): lets tests confirm the in-flight-jump split case (see
# `_BacktrackRange`) is actually being exercised, not just the simpler
# node-split case.
SPLIT_STATS = {'node': 0, 'in_flight': 0}


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


def BackwardFrameDp(D, c0, c1):
  """B[i,j] = min cost from (i,j) to (N,M) - the match/jump costs are
  symmetric under reversal, so this is just `FastFrameDp` run on the
  reversed problem."""
  D_rev = D[::-1, ::-1]
  F_rev = FastFrameDp(D_rev, c0, c1)
  return F_rev[::-1, ::-1]


def CrossingCost(F, B, c0, c1, i_mid):
  """How the optimal path crosses row `i_mid`, as three length-`M+1`
  arrays (one per column `j`): `node_cost` (the path has an actual node
  at `(i_mid,j)`) and `in_flight_cost` (an i-jump that started before
  `i_mid` and lands after it, without ever touching row `i_mid`) - see
  the module docstring's point about why a flat-cost jump needs this
  second case. `crossing_cost` is their minimum, i.e. the true cost of
  passing through column `j` at row `i_mid` by either means.

  `in_flight_cost(j) = (c0 + min_{i'<i_mid} [F(i',j) - c1*i'])
                        + min_{i''>=i_mid} [B(i'',j) + c1*i'']`

  - the same `ColMinShifted` quantity `FastFrameDp` already computes
  while sweeping forward, paired with its mirror-image shifted suffix
  minimum of the backward values."""
  N, Mp1 = F.shape[0] - 1, F.shape[1]
  node_cost = F[i_mid, :] + B[i_mid, :]

  if i_mid > 0:
    shifted_F = F[:i_mid, :] - c1 * np.arange(i_mid, dtype=np.float64)[:, None]
    col_min_shifted = shifted_F.min(axis=0)
  else:
    col_min_shifted = np.full(Mp1, np.inf)
  credit0 = c0 + col_min_shifted

  shifted_B = B[i_mid:, :] + c1 * np.arange(i_mid, N + 1,
                                            dtype=np.float64)[:, None]
  bcredit_shifted = shifted_B.min(axis=0)
  in_flight_cost = credit0 + bcredit_shifted

  crossing_cost = np.minimum(node_cost, in_flight_cost)
  return crossing_cost, node_cost, in_flight_cost


def _BruteForceBacktrack(D, c0, c1, i_offset, j_offset):
  """Base case: O(N*M*(N+M)) backtrack with full backpointers - only for
  small sub-grids. Returns a list of `(move, i0, j0, i1, j1)` segments in
  *global* coordinates (`i_offset`/`j_offset` added back in); `move` is
  one of `'diag'`, `'ijump'`, `'jjump'`."""
  N, M = D.shape
  F = np.full((N + 1, M + 1), np.inf)
  back = [[None] * (M + 1) for _ in range(N + 1)]
  F[0, 0] = 0.0
  for i in range(N + 1):
    for j in range(M + 1):
      if i == 0 and j == 0:
        continue
      best, move = np.inf, None
      if i > 0 and j > 0:
        v = F[i - 1, j - 1] + D[i - 1, j - 1]
        if v < best:
          best, move = v, ('diag', i - 1, j - 1)
      if i > 0:
        ip = np.arange(i)
        vals = F[:i, j] + c0 + c1 * (i - ip)
        k = int(np.argmin(vals))
        if vals[k] < best:
          best, move = vals[k], ('ijump', k, j)
      if j > 0:
        jp = np.arange(j)
        vals = F[i, :j] + c0 + c1 * (j - jp)
        k = int(np.argmin(vals))
        if vals[k] < best:
          best, move = vals[k], ('jjump', i, k)
      F[i, j] = best
      back[i][j] = move

  path = []
  i, j = N, M
  while (i, j) != (0, 0):
    move, pi, pj = back[i][j]
    path.append((move, pi + i_offset, pj + j_offset, i + i_offset,
                j + j_offset))
    i, j = pi, pj
  path.reverse()
  return path


def Backtrack(D, c0, c1, base_size=40):
  """Returns the optimal path from `(0,0)` to `(N,M)` as a list of
  `(move, i0, j0, i1, j1)` segments, `move` one of `'diag'`, `'ijump'`,
  `'jjump'`. Linear memory (modulo the base case, which is intentionally
  small) via Hirschberg's divide-and-conquer: recomputes `FastFrameDp`/
  `BackwardFrameDp` on shrinking sub-grids, so it's O(N*M*log(min(N,M)))
  time - fine for validation on small/medium inputs; see
  `StreamingBacktrack` for the GPU version that avoids both the full
  matrices and the log factor at whole-film scale."""
  _CheckDpParams(c0, c1)
  return _BacktrackRange(D, c0, c1, 0, D.shape[0], 0, D.shape[1], base_size)


def _BacktrackRange(D, c0, c1, i_lo, i_hi, j_lo, j_hi, base_size):
  N_sub, M_sub = i_hi - i_lo, j_hi - j_lo
  if N_sub <= base_size or M_sub <= base_size:
    return _BruteForceBacktrack(D[i_lo:i_hi, j_lo:j_hi], c0, c1, i_lo, j_lo)

  i_mid_local = N_sub // 2
  sub_D = D[i_lo:i_hi, j_lo:j_hi]
  F = FastFrameDp(sub_D, c0, c1)
  B = BackwardFrameDp(sub_D, c0, c1)
  crossing_cost, node_cost, in_flight_cost = CrossingCost(
      F, B, c0, c1, i_mid_local)
  j_star_local = int(np.argmin(crossing_cost))
  j_star = j_lo + j_star_local
  i_mid = i_lo + i_mid_local

  if node_cost[j_star_local] <= in_flight_cost[j_star_local]:
    SPLIT_STATS['node'] += 1
    left = _BacktrackRange(D, c0, c1, i_lo, i_mid, j_lo, j_star, base_size)
    right = _BacktrackRange(D, c0, c1, i_mid, i_hi, j_star, j_hi, base_size)
    return left + right

  SPLIT_STATS['in_flight'] += 1

  # The in-flight i-jump spans i_mid - recover its true source/
  # destination rows (not i_mid itself) so the two halves don't each pay
  # c0 for "their half" of what is really a single jump.
  shifted_F = F[:i_mid_local, :] - c1 * np.arange(
      i_mid_local, dtype=np.float64)[:, None]
  i_source_local = int(np.argmin(shifted_F[:, j_star_local]))
  i_source = i_lo + i_source_local
  shifted_B = B[i_mid_local:, :] + c1 * np.arange(
      i_mid_local, N_sub + 1, dtype=np.float64)[:, None]
  i_dest_local = i_mid_local + int(np.argmin(shifted_B[:, j_star_local]))
  i_dest = i_lo + i_dest_local

  left = _BacktrackRange(D, c0, c1, i_lo, i_source, j_lo, j_star, base_size)
  jump_segment = [('ijump', i_source, j_star, i_dest, j_star)]
  right = _BacktrackRange(D, c0, c1, i_dest, i_hi, j_star, j_hi, base_size)
  return left + jump_segment + right


def PathCost(path, D, c0, c1):
  """Recomputes a path's total cost directly from its segments - a
  self-contained sanity check independent of the DP machinery that
  produced it."""
  total = 0.0
  for move, i0, j0, i1, j1 in path:
    if move == 'diag':
      total += D[i0, j0]
    elif move in ('ijump', 'jjump'):
      length = (i1 - i0) + (j1 - j0)
      total += c0 + c1 * length
    else:
      raise ValueError(f'unknown move {move}')
  return total


def PathEndpoints(path, N, M):
  """Checks a path's segments chain continuously from `(0,0)` to
  `(N,M)` - returns `(True, 'ok')` or `(False, <message>)`, for use in
  tests."""
  i, j = 0, 0
  for move, i0, j0, i1, j1 in path:
    if (i0, j0) != (i, j):
      return False, f'gap: expected to start at {(i,j)}, segment starts at {(i0,j0)}'
    i, j = i1, j1
  if (i, j) != (N, M):
    return False, f'path ends at {(i,j)}, expected {(N,M)}'
  return True, 'ok'
