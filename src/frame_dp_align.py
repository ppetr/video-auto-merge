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
import functools

import jax
import jax.numpy as jnp
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


# --- GPU/streaming path: usable at whole-film scale -------------------
#
# `Backtrack` above recomputes full (N+1,M+1) `FastFrameDp`/
# `BackwardFrameDp` matrices at every recursion level - fine for
# validation, but neither the time (an extra log factor) nor the memory
# (the full matrix) works at whole-film scale (hundreds of thousands of
# frames per side). The functions below replace both with chunked GPU
# sweeps that only ever materialize one row's cost block at a time.


def _HardRowStep(F_prev, col_min_shifted, i, D_row, c0, c1, a0_override=None):
  """One row of the `FastFrameDp` recurrence, as JAX array ops - same
  math as the CPU version's loop body, so it composes into
  `jax.lax.scan` for a whole chunk of rows at once (see `_ChunkedSweep`).

  `a0_override` patches in the true `F(0,0)=0` base case (row 0 has no
  real predecessor row) - it must be applied to `A` itself, *before* the
  prefix-min below, not patched onto the final row afterward: every
  later column's value depends on `A[0]` through that scan."""
  M = F_prev.shape[0] - 1
  j_idx = jnp.arange(M + 1, dtype=jnp.float32)

  diag = jnp.concatenate([jnp.array([jnp.inf]), F_prev[:-1] + D_row])
  ijump = c0 + c1 * i + col_min_shifted
  A = jnp.minimum(diag, ijump)
  if a0_override is not None:
    A = A.at[0].set(a0_override)

  B = A - c1 * j_idx
  R = jax.lax.cummin(B)
  R_shift = jnp.concatenate([jnp.array([jnp.inf]), R[:-1]])
  F_cur = jnp.minimum(A, c0 + c1 * j_idx + R_shift)
  new_col_min_shifted = jnp.minimum(col_min_shifted, F_cur - c1 * i)
  return F_cur, new_col_min_shifted


def _MakeRunChunk():
  """Builds the single, module-level-cached jit function used for every
  chunk of rows, with `c0`/`c1` as traced (not closed-over) arguments -
  so JAX compiles it once per distinct `(chunk_len, M)` shape and reuses
  that executable across an entire Hirschberg recursion, no matter how
  many times `StreamingBacktrack` calls back in.

  Besides `F`/`col_min_shifted`, this also tracks `src`: the row index
  that *achieves* the running `col_min_shifted` for each column - needed
  to recover an in-flight jump's true start/end row without ever
  materializing the full matrix (see `StreamingBacktrack`).

  This must NOT be a closure created fresh inside `_ChunkedSweep` on
  every call - a divide-and-conquer recursion calls back into this
  hundreds of times, and a freshly-compiled executable per call
  accumulates on the GPU for as long as anything keeps a reference to
  it, exhausting a few GB of VRAM long before any estimate based on
  actual data volume would predict (confirmed as the real cause of a
  `RESOURCE_EXHAUSTED` error that plain data-size math said shouldn't
  happen)."""

  @functools.partial(jax.jit, static_argnums=(7,))
  def RunChunk(F, cms, src, i0, D_chunk, c0, c1, M):
    del M  # only needed to pin the jit cache key to this chunk's shape.

    def step(carry, D_row):
      F_prev, cms, src, i = carry
      F_cur, new_cms = _HardRowStep(F_prev, cms, i, D_row, c0, c1)
      improved = new_cms < cms
      new_src = jnp.where(improved, i, src)
      return (F_cur, new_cms, new_src, i + 1.0), None

    (F_final, cms_final, src_final, i_final), _ = jax.lax.scan(
        step, (F, cms, src, i0), D_chunk)
    return F_final, cms_final, src_final, i_final

  return RunChunk


_RUN_CHUNK = None


def _GetRunChunk():
  global _RUN_CHUNK
  if _RUN_CHUNK is None:
    _RUN_CHUNK = _MakeRunChunk()
  return _RUN_CHUNK


def _ChunkedSweep(sig, right_cost, c0, c1, stop_row, chunk):
  """Runs the forward sweep through `stop_row` frames of `sig`, tracking
  `col_min_shifted` *and* which row achieves it for each column. Returns
  `(F_at_stop_row, cms_before_last, src_before_last, cms_after_last,
  src_after_last)` - both snapshots, since the forward and backward
  checkpoints need opposite ones (see `ForwardCheckpoint`/
  `BackwardCheckpoint`).

  `sig`/`right_cost` are expected as plain numpy arrays, not already on
  device - only `right_cost` (this sub-range's size) and one
  `chunk`-sized slice of `sig` at a time are moved to the device, both
  freed when this function returns. Hirschberg's recursion holds many
  overlapping sub-range slices alive on the call stack at once; if the
  *caller* kept its encodings as device arrays and this function sliced
  them, each outstanding stack frame would pin a distinct device-side
  copy for as long as it's live, rather than the cheap host-side numpy
  view this relies on instead (confirmed as the real cause of an
  out-of-memory failure on an input that comfortably fits on the GPU by
  itself - not a data-volume problem, a reference-lifetime one)."""
  M = right_cost.shape[0]
  run_chunk = _GetRunChunk()
  right_cost_dev = jnp.asarray(right_cost)

  F0 = jnp.full(M + 1, jnp.inf, dtype=jnp.float32)
  cms0 = jnp.full(M + 1, jnp.inf, dtype=jnp.float32)
  src0 = jnp.zeros(M + 1, dtype=jnp.float32)  # value irrelevant - cms0 is
  # +inf so row 0 always "improves", overwriting src regardless.
  dummy = jnp.zeros(M, dtype=jnp.float32)
  F, cms = _HardRowStep(F0, cms0, 0.0, dummy, c0, c1, a0_override=0.0)
  src = jnp.where(cms < cms0, 0.0, src0)
  i = jnp.float32(1.0)

  if stop_row == 0:
    return F, cms0, src0, cms, src

  start = 0
  while start < stop_row - 1:
    end = min(start + chunk, stop_row - 1)
    D_chunk = jnp.asarray(sig[start:end]) @ right_cost_dev.T
    F, cms, src, i = run_chunk(F, cms, src, i, D_chunk, c0, c1, M)
    start = end

  cms_before_last, src_before_last = cms, src
  D_last = jnp.asarray(sig[stop_row - 1:stop_row]) @ right_cost_dev.T
  F, cms, src, i = run_chunk(F, cms, src, i, D_last, c0, c1, M)
  return F, cms_before_last, src_before_last, cms, src


def ForwardCheckpoint(left_onehot, right_cost, c0, c1, i_mid, chunk=300):
  """`F(i_mid,:)`, `ColMinShifted_{i_mid}(:)` and its source row (rows
  `0..i_mid-1`) - the forward half of what the Hirschberg split at
  `i_mid` needs, without materializing the full matrix."""
  F, cms_before, src_before, _, _ = _ChunkedSweep(left_onehot, right_cost,
                                                  c0, c1, i_mid, chunk)
  return np.array(F), np.array(cms_before), np.array(src_before)


def BackwardCheckpoint(left_onehot_rev, right_cost_rev, c0, c1, N, M, i_mid,
                       chunk=300):
  """`B(i_mid,:)`, `BCreditShifted_{i_mid}(:) = min_{i''>=i_mid}
  [B(i'',j)+c1*i'']` and its source row (in *original* coordinates), via
  the reversed sweep: running `_ChunkedSweep` on the reversed sequences
  gives `F'(k,j') = B(N-k,M-j')`, so `B(i_mid,:)` is `F'` at row
  `k=N-i_mid`.

  `BCreditShifted`/its source row are NOT simply the reversed sweep's
  own `ColMinShifted`/`src` at that point - the index reversal needs an
  extra shift to undo, re-derived here (and checked numerically against
  the CPU-exact `BackwardFrameDp`/`CrossingCost` before being trusted)
  as:

    BCreditShifted_{i_mid}(j) = ColMinShifted'_{N-i_mid+1}(M-j) + c1*N
    i_dest(j) [the row achieving it] = N - source'_{N-i_mid+1}(M-j)

  i.e. run the reversed sweep *through* row `k=N-i_mid` (inclusive, not
  "before" it as the forward checkpoint does), take its
  `ColMinShifted`/source snapshot from *after* that row, then reverse
  the column axis, shift the value by `+c1*N` and convert the row index
  via `N-k'`.

  `left_onehot_rev`/`right_cost_rev` must already be the *reversed*
  sequences (reverse frame order on both sides before building the
  cost-ready encoding - reversing the encoded arrays afterward would not
  reverse the underlying cost matrix correctly)."""
  k = N - i_mid
  F_rev, _, _, cms_after, src_after = _ChunkedSweep(left_onehot_rev,
                                                    right_cost_rev, c0, c1,
                                                    k, chunk)
  B_mid = np.array(F_rev)[::-1]
  bcredit_shifted = np.array(cms_after)[::-1] + c1 * N
  i_dest = N - np.array(src_after)[::-1]
  return B_mid, bcredit_shifted, i_dest


def StreamingBacktrack(left_onehot,
                       right_cost,
                       left_onehot_rev,
                       right_cost_rev,
                       c0,
                       c1,
                       N_total,
                       M_total,
                       i_lo,
                       i_hi,
                       j_lo,
                       j_hi,
                       base_size=500,
                       chunk=300):
  """Same contract as `Backtrack`, but using the GPU streaming
  checkpoints above instead of full-matrix `FastFrameDp`/
  `BackwardFrameDp` - usable at whole-film scale (only the base case
  ever materializes a small cost sub-block).

  `left_onehot`/`right_cost` and their `_rev` counterparts must be the
  cost-ready encodings of the *entire* original sequences (and their
  full reversals) - the same arrays at every recursion level; slicing to
  the current sub-range happens internally. `N_total`/`M_total` are the
  full sequence lengths (needed to correctly slice the reversed arrays
  for a sub-range)."""
  N_sub, M_sub = i_hi - i_lo, j_hi - j_lo
  if N_sub <= base_size or M_sub <= base_size:
    D_sub = np.array(left_onehot[i_lo:i_hi] @ right_cost[j_lo:j_hi].T)
    return _BruteForceBacktrack(D_sub, c0, c1, i_lo, j_lo)

  i_mid_local = N_sub // 2
  i_mid = i_lo + i_mid_local

  F_mid, cms_before, src_before = ForwardCheckpoint(
      left_onehot[i_lo:i_hi], right_cost[j_lo:j_hi], c0, c1, i_mid_local,
      chunk)
  # Reversed sub-range of [i_lo,i_hi)/[j_lo,j_hi) within the full
  # reversed arrays.
  left_rev_sub = left_onehot_rev[N_total - i_hi:N_total - i_lo]
  right_rev_sub = right_cost_rev[M_total - j_hi:M_total - j_lo]
  B_mid, bcredit, i_dest_full = BackwardCheckpoint(left_rev_sub,
                                                   right_rev_sub, c0, c1,
                                                   N_sub, M_sub,
                                                   i_mid_local, chunk)
  # i_dest_full came back in [i_lo,i_hi)-local coordinates (N_sub was
  # passed as the local length) - shift to global, same as src_before.
  i_dest_full = i_dest_full + i_lo
  src_before_full = src_before + i_lo

  node_cost = F_mid + B_mid
  in_flight_cost = c0 + cms_before + bcredit
  crossing_cost = np.minimum(node_cost, in_flight_cost)
  j_star_local = int(np.argmin(crossing_cost))
  j_star = j_lo + j_star_local

  if node_cost[j_star_local] <= in_flight_cost[j_star_local]:
    SPLIT_STATS['node'] += 1
    left = StreamingBacktrack(left_onehot, right_cost, left_onehot_rev,
                              right_cost_rev, c0, c1, N_total, M_total,
                              i_lo, i_mid, j_lo, j_star, base_size, chunk)
    right = StreamingBacktrack(left_onehot, right_cost, left_onehot_rev,
                               right_cost_rev, c0, c1, N_total, M_total,
                               i_mid, i_hi, j_star, j_hi, base_size, chunk)
    return left + right

  SPLIT_STATS['in_flight'] += 1
  # The in-flight i-jump's true source/destination rows, recovered
  # directly from the checkpoints' source-row tracking (no full matrix
  # needed) - same reasoning as `_BacktrackRange`'s node_cost/
  # in_flight_cost comparison above, just without materializing F/B.
  i_source = int(src_before_full[j_star_local])
  i_dest = int(i_dest_full[j_star_local])

  left = StreamingBacktrack(left_onehot, right_cost, left_onehot_rev,
                            right_cost_rev, c0, c1, N_total, M_total, i_lo,
                            i_source, j_lo, j_star, base_size, chunk)
  jump_segment = [('ijump', i_source, j_star, i_dest, j_star)]
  right = StreamingBacktrack(left_onehot, right_cost, left_onehot_rev,
                             right_cost_rev, c0, c1, N_total, M_total,
                             i_dest, i_hi, j_star, j_hi, base_size, chunk)
  return left + jump_segment + right
