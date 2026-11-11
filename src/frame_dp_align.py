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

  - match: `(i,j) -> (i+1,j+1)`, cost `cost_matrix[i,j]` (the two frames'
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
(`StreamingBacktrack`) usable on whole-film-length inputs. `AlignFrames`
is the public entry point tying all of that together.
"""
import dataclasses
import functools
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

# Diagnostic counters (not thread-safe - fine for single-process tests/
# scripts): lets tests confirm the in-flight-jump split case (see
# `_BacktrackRange`) is actually being exercised, not just the simpler
# node-split case.
SPLIT_STATS = {'node': 0, 'in_flight': 0}


def _CheckDpParams(c0, c1):
  if c0 < 0 or c1 < 0:
    raise ValueError(f'c0 and c1 must be non-negative (got c0={c0}, c1={c1})')


def NaiveFrameDp(cost_matrix, c0, c1):
  """`cost_to[i,j]` = min cost of a path from `(0,0)` to `(i,j)`.
  O(N*M*(N+M)): every candidate jump origin is scanned explicitly, no
  shortcut - reference implementation for small inputs only, to
  validate `FastFrameDp` against."""
  _CheckDpParams(c0, c1)
  n, m = cost_matrix.shape
  cost_to = np.full((n + 1, m + 1), np.inf)
  cost_to[0, 0] = 0.0
  for i in range(n + 1):
    for j in range(m + 1):
      if i == 0 and j == 0:
        continue
      best = np.inf
      if i > 0 and j > 0:
        best = min(best, cost_to[i - 1, j - 1] + cost_matrix[i - 1, j - 1])
      if i > 0:
        ip = np.arange(i)
        best = min(best, float((cost_to[:i, j] + c0 + c1 * (i - ip)).min()))
      if j > 0:
        jp = np.arange(j)
        best = min(best, float((cost_to[i, :j] + c0 + c1 * (j - jp)).min()))
      cost_to[i, j] = best
  return cost_to


def FastFrameDp(cost_matrix, c0, c1):
  """Same recurrence as `NaiveFrameDp`, O(N*M): the "jump from anywhere
  before the current position" term collapses into a running prefix
  minimum rather than an explicit per-cell scan over every candidate
  origin.

  Derivation, for one row `i`: let `col_min_shifted(j) = min_{i'<i}
  [cost_to(i',j) - c1*i']` (maintained incrementally across rows - the
  cross-row jump term for row `i` is then `c0 + c1*i +
  col_min_shifted(j)`, i.e. no re-scanning of earlier rows). Combine
  that with the match term into `candidate(j)`. The within-row jump
  term needs `min_{j'<j} cost_to(i,j')`, but since a jump's cost is
  non-negative, `prefix(j) := min_{j'<=j} cost_to(i,j')` satisfies
  `prefix(j) = min(candidate(j), prefix(j-1))` - i.e. `prefix` is simply
  the prefix minimum of `candidate` shifted by `c1*j` (same shift trick
  as the cross-row term), not a new dependency on `cost_to` itself. So
  the whole row reduces to one forward pass over `candidate`
  (`np.minimum.accumulate`), not a sequential per-column loop - the same
  shape of computation `jax.lax.cummin`/`jax.lax.scan` run on a GPU (see
  `_HardRowStep`).
  """
  _CheckDpParams(c0, c1)
  n, m = cost_matrix.shape
  j_idx = np.arange(m + 1, dtype=np.float64)

  f_prev = np.full(m + 1, np.inf)
  col_min_shifted = np.full(m + 1, np.inf)
  f = np.zeros((n + 1, m + 1))

  for i in range(n + 1):
    diag = np.full(m + 1, np.inf)
    diag[1:] = f_prev[:-1] + (cost_matrix[i - 1, :] if i > 0 else 0.0)
    ijump = c0 + c1 * i + col_min_shifted
    candidate = np.minimum(diag, ijump)
    if i == 0:
      candidate[0] = 0.0  # true base case - no incoming edge at all.

    shifted = candidate - c1 * j_idx
    prefix_min = np.minimum.accumulate(shifted)
    prefix_min_shift = np.concatenate([[np.inf], prefix_min[:-1]])
    f_cur = np.minimum(candidate, c0 + c1 * j_idx + prefix_min_shift)

    f[i] = f_cur
    col_min_shifted = np.minimum(col_min_shifted, f_cur - c1 * i)
    f_prev = f_cur

  return f


def BackwardFrameDp(cost_matrix, c0, c1):
  """`cost_from[i,j]` = min cost from `(i,j)` to `(N,M)` - the match/
  jump costs are symmetric under reversal, so this is just
  `FastFrameDp` run on the reversed problem."""
  cost_matrix_rev = cost_matrix[::-1, ::-1]
  cost_rev = FastFrameDp(cost_matrix_rev, c0, c1)
  return cost_rev[::-1, ::-1]


def CrossingCost(cost_to, cost_from, c0, c1, i_mid):
  """How the optimal path crosses row `i_mid`, as three length-`M+1`
  arrays (one per column `j`): `node_cost` (the path has an actual node
  at `(i_mid,j)`) and `in_flight_cost` (an i-jump that started before
  `i_mid` and lands after it, without ever touching row `i_mid`) - see
  the module docstring's point about why a flat-cost jump needs this
  second case. `crossing_cost` is their minimum, i.e. the true cost of
  passing through column `j` at row `i_mid` by either means.

  `in_flight_cost(j) = (c0 + min_{i'<i_mid} [cost_to(i',j) - c1*i'])
                        + min_{i''>=i_mid} [cost_from(i'',j) + c1*i'']`

  - the same `col_min_shifted` quantity `FastFrameDp` already computes
  while sweeping forward, paired with its mirror-image shifted suffix
  minimum of the backward values."""
  n, mp1 = cost_to.shape[0] - 1, cost_to.shape[1]
  node_cost = cost_to[i_mid, :] + cost_from[i_mid, :]

  if i_mid > 0:
    shifted = (cost_to[:i_mid, :] -
              c1 * np.arange(i_mid, dtype=np.float64)[:, None])
    col_min_shifted = shifted.min(axis=0)
  else:
    col_min_shifted = np.full(mp1, np.inf)
  credit0 = c0 + col_min_shifted

  shifted_back = cost_from[i_mid:, :] + c1 * np.arange(
      i_mid, n + 1, dtype=np.float64)[:, None]
  bcredit_shifted = shifted_back.min(axis=0)
  in_flight_cost = credit0 + bcredit_shifted

  crossing_cost = np.minimum(node_cost, in_flight_cost)
  return crossing_cost, node_cost, in_flight_cost


def _BruteForceBacktrack(cost_matrix, c0, c1, i_offset, j_offset):
  """Base case: O(N*M*(N+M)) backtrack with full backpointers - only for
  small sub-grids. Returns a list of `(move, i0, j0, i1, j1)` segments
  in *global* coordinates (`i_offset`/`j_offset` added back in); `move`
  is one of `'diag'`, `'ijump'`, `'jjump'`."""
  n, m = cost_matrix.shape
  cost_to = np.full((n + 1, m + 1), np.inf)
  back = [[None] * (m + 1) for _ in range(n + 1)]
  cost_to[0, 0] = 0.0
  for i in range(n + 1):
    for j in range(m + 1):
      if i == 0 and j == 0:
        continue
      best, move = np.inf, None
      if i > 0 and j > 0:
        v = cost_to[i - 1, j - 1] + cost_matrix[i - 1, j - 1]
        if v < best:
          best, move = v, ('diag', i - 1, j - 1)
      if i > 0:
        ip = np.arange(i)
        vals = cost_to[:i, j] + c0 + c1 * (i - ip)
        k = int(np.argmin(vals))
        if vals[k] < best:
          best, move = vals[k], ('ijump', k, j)
      if j > 0:
        jp = np.arange(j)
        vals = cost_to[i, :j] + c0 + c1 * (j - jp)
        k = int(np.argmin(vals))
        if vals[k] < best:
          best, move = vals[k], ('jjump', i, k)
      cost_to[i, j] = best
      back[i][j] = move

  path = []
  i, j = n, m
  while (i, j) != (0, 0):
    move, pi, pj = back[i][j]
    path.append((move, pi + i_offset, pj + j_offset, i + i_offset,
                j + j_offset))
    i, j = pi, pj
  path.reverse()
  return path


def Backtrack(cost_matrix, c0, c1, base_size=40):
  """Returns the optimal path from `(0,0)` to `(N,M)` as a list of
  `(move, i0, j0, i1, j1)` segments, `move` one of `'diag'`, `'ijump'`,
  `'jjump'`. Linear memory (modulo the base case, which is
  intentionally small) via Hirschberg's divide-and-conquer: recomputes
  `FastFrameDp`/`BackwardFrameDp` on shrinking sub-grids, so it's
  O(N*M*log(min(N,M))) time - fine for validation on small/medium
  inputs; see `StreamingBacktrack` for the GPU version that avoids both
  the full matrices and the log factor at whole-film scale."""
  _CheckDpParams(c0, c1)
  return _BacktrackRange(cost_matrix, c0, c1, 0, cost_matrix.shape[0], 0,
                        cost_matrix.shape[1], base_size)


def _BacktrackRange(cost_matrix, c0, c1, i_lo, i_hi, j_lo, j_hi, base_size):
  n_sub, m_sub = i_hi - i_lo, j_hi - j_lo
  if n_sub <= base_size or m_sub <= base_size:
    return _BruteForceBacktrack(cost_matrix[i_lo:i_hi, j_lo:j_hi], c0, c1,
                                i_lo, j_lo)

  i_mid_local = n_sub // 2
  sub_cost = cost_matrix[i_lo:i_hi, j_lo:j_hi]
  cost_to = FastFrameDp(sub_cost, c0, c1)
  cost_from = BackwardFrameDp(sub_cost, c0, c1)
  crossing_cost, node_cost, in_flight_cost = CrossingCost(
      cost_to, cost_from, c0, c1, i_mid_local)
  j_star_local = int(np.argmin(crossing_cost))
  j_star = j_lo + j_star_local
  i_mid = i_lo + i_mid_local

  if node_cost[j_star_local] <= in_flight_cost[j_star_local]:
    SPLIT_STATS['node'] += 1
    left = _BacktrackRange(cost_matrix, c0, c1, i_lo, i_mid, j_lo, j_star,
                           base_size)
    right = _BacktrackRange(cost_matrix, c0, c1, i_mid, i_hi, j_star, j_hi,
                            base_size)
    return left + right

  SPLIT_STATS['in_flight'] += 1

  # The in-flight i-jump spans i_mid - recover its true source/
  # destination rows (not i_mid itself) so the two halves don't each pay
  # c0 for "their half" of what is really a single jump.
  shifted_to = cost_to[:i_mid_local, :] - c1 * np.arange(
      i_mid_local, dtype=np.float64)[:, None]
  i_source_local = int(np.argmin(shifted_to[:, j_star_local]))
  i_source = i_lo + i_source_local
  shifted_from = cost_from[i_mid_local:, :] + c1 * np.arange(
      i_mid_local, n_sub + 1, dtype=np.float64)[:, None]
  i_dest_local = i_mid_local + int(np.argmin(shifted_from[:, j_star_local]))
  i_dest = i_lo + i_dest_local

  left = _BacktrackRange(cost_matrix, c0, c1, i_lo, i_source, j_lo, j_star,
                        base_size)
  jump_segment = [('ijump', i_source, j_star, i_dest, j_star)]
  right = _BacktrackRange(cost_matrix, c0, c1, i_dest, i_hi, j_star, j_hi,
                          base_size)
  return left + jump_segment + right


def PathCost(path, cost_matrix, c0, c1):
  """Recomputes a path's total cost directly from its segments - a
  self-contained sanity check independent of the DP machinery that
  produced it."""
  total = 0.0
  for move, i0, j0, i1, j1 in path:
    if move == 'diag':
      total += cost_matrix[i0, j0]
    elif move in ('ijump', 'jjump'):
      length = (i1 - i0) + (j1 - j0)
      total += c0 + c1 * length
    else:
      raise ValueError(f'unknown move {move}')
  return total


def PathEndpoints(path, n, m):
  """Checks a path's segments chain continuously from `(0,0)` to
  `(N,M)` - returns `(True, 'ok')` or `(False, <message>)`, for use in
  tests."""
  i, j = 0, 0
  for _, i0, j0, i1, j1 in path:
    if (i0, j0) != (i, j):
      return False, (f'gap: expected to start at {(i,j)}, segment starts '
                     f'at {(i0,j0)}')
    i, j = i1, j1
  if (i, j) != (n, m):
    return False, f'path ends at {(i,j)}, expected {(n,m)}'
  return True, 'ok'


# --- GPU/streaming path: usable at whole-film scale -------------------
#
# `Backtrack` above recomputes full (N+1,M+1) `FastFrameDp`/
# `BackwardFrameDp` matrices at every recursion level - fine for
# validation, but neither the time (an extra log factor) nor the memory
# (the full matrix) works at whole-film scale (hundreds of thousands of
# frames per side). The functions below replace both with chunked GPU
# sweeps that only ever materialize one row's cost block at a time.


def _HardRowStep(f_prev, col_min_shifted, i, d_row, c0, c1, a0_override=None):
  """One row of the `FastFrameDp` recurrence, as JAX array ops - same
  math as the CPU version's loop body, so it composes into
  `jax.lax.scan` for a whole chunk of rows at once (see `_ChunkedSweep`).

  `a0_override` patches in the true `cost_to(0,0)=0` base case (row 0
  has no real predecessor row) - it must be applied to the row's
  candidate costs themselves, *before* the prefix-min below, not
  patched onto the final row afterward: every later column's value
  depends on column 0 through that scan."""
  m = f_prev.shape[0] - 1
  j_idx = jnp.arange(m + 1, dtype=jnp.float32)

  diag = jnp.concatenate([jnp.array([jnp.inf]), f_prev[:-1] + d_row])
  ijump = c0 + c1 * i + col_min_shifted
  candidate = jnp.minimum(diag, ijump)
  if a0_override is not None:
    candidate = candidate.at[0].set(a0_override)

  shifted = candidate - c1 * j_idx
  prefix_min = jax.lax.cummin(shifted)
  prefix_min_shift = jnp.concatenate([jnp.array([jnp.inf]), prefix_min[:-1]])
  f_cur = jnp.minimum(candidate, c0 + c1 * j_idx + prefix_min_shift)
  new_col_min_shifted = jnp.minimum(col_min_shifted, f_cur - c1 * i)
  return f_cur, new_col_min_shifted


def _MakeRunChunk():
  """Builds the single, module-level-cached jit function used for every
  chunk of rows, with `c0`/`c1` as traced (not closed-over) arguments -
  so JAX compiles it once per distinct `(chunk_len, M)` shape and reuses
  that executable across an entire Hirschberg recursion, no matter how
  many times `StreamingBacktrack` calls back in.

  Besides the cost row and `col_min_shifted`, this also tracks `src`:
  the row index that *achieves* the running `col_min_shifted` for each
  column - needed to recover an in-flight jump's true start/end row
  without ever materializing the full matrix (see
  `StreamingBacktrack`).

  This must NOT be a closure created fresh inside `_ChunkedSweep` on
  every call - a divide-and-conquer recursion calls back into this
  hundreds of times, and a freshly-compiled executable per call
  accumulates on the GPU for as long as anything keeps a reference to
  it, exhausting a few GB of VRAM long before any estimate based on
  actual data volume would predict (confirmed as the real cause of a
  `RESOURCE_EXHAUSTED` error that plain data-size math said shouldn't
  happen)."""

  @functools.partial(jax.jit, static_argnums=(7,))
  def RunChunk(f, cms, src, i0, d_chunk, c0, c1, m):
    del m  # only needed to pin the jit cache key to this chunk's shape.

    def Step(carry, d_row):
      f_prev, cms, src, i = carry
      f_cur, new_cms = _HardRowStep(f_prev, cms, i, d_row, c0, c1)
      improved = new_cms < cms
      new_src = jnp.where(improved, i, src)
      return (f_cur, new_cms, new_src, i + 1.0), None

    (f_final, cms_final, src_final, i_final), _ = jax.lax.scan(
        Step, (f, cms, src, i0), d_chunk)
    return f_final, cms_final, src_final, i_final

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
  `(f_at_stop_row, cms_before_last, src_before_last, cms_after_last,
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
  m = right_cost.shape[0]
  run_chunk = _GetRunChunk()
  right_cost_dev = jnp.asarray(right_cost)

  f0 = jnp.full(m + 1, jnp.inf, dtype=jnp.float32)
  cms0 = jnp.full(m + 1, jnp.inf, dtype=jnp.float32)
  src0 = jnp.zeros(m + 1, dtype=jnp.float32)  # value irrelevant - cms0 is
  # +inf so row 0 always "improves", overwriting src regardless.
  dummy = jnp.zeros(m, dtype=jnp.float32)
  f, cms = _HardRowStep(f0, cms0, 0.0, dummy, c0, c1, a0_override=0.0)
  src = jnp.where(cms < cms0, 0.0, src0)
  i = jnp.float32(1.0)

  if stop_row == 0:
    return f, cms0, src0, cms, src

  start = 0
  while start < stop_row - 1:
    end = min(start + chunk, stop_row - 1)
    d_chunk = jnp.asarray(sig[start:end]) @ right_cost_dev.T
    f, cms, src, i = run_chunk(f, cms, src, i, d_chunk, c0, c1, m)
    start = end

  cms_before_last, src_before_last = cms, src
  d_last = jnp.asarray(sig[stop_row - 1:stop_row]) @ right_cost_dev.T
  f, cms, src, i = run_chunk(f, cms, src, i, d_last, c0, c1, m)
  return f, cms_before_last, src_before_last, cms, src


def ForwardCheckpoint(left_onehot, right_cost, c0, c1, i_mid, chunk=300):
  """`cost_to(i_mid,:)`, `col_min_shifted` at `i_mid` (aggregating rows
  `0..i_mid-1`) and its source row - the forward half of what the
  Hirschberg split at `i_mid` needs, without materializing the full
  matrix."""
  f, cms_before, src_before, _, _ = _ChunkedSweep(left_onehot, right_cost,
                                                  c0, c1, i_mid, chunk)
  return np.array(f), np.array(cms_before), np.array(src_before)


def BackwardCheckpoint(left_onehot_rev, right_cost_rev, c0, c1, n, m, i_mid,
                       chunk=300):
  """`cost_from(i_mid,:)`, `bcredit_shifted(i_mid,:) = min_{i''>=i_mid}
  [cost_from(i'',j)+c1*i'']` and its source row (in *original*
  coordinates), via the reversed sweep: running `_ChunkedSweep` on the
  reversed sequences gives `cost_rev(k,j') = cost_from(N-k,M-j')`, so
  `cost_from(i_mid,:)` is `cost_rev` at row `k=N-i_mid`.

  `bcredit_shifted`/its source row are NOT simply the reversed sweep's
  own `col_min_shifted`/`src` at that point - the index reversal needs
  an extra shift to undo, re-derived here (and checked numerically
  against the CPU-exact `BackwardFrameDp`/`CrossingCost` before being
  trusted) as:

    bcredit_shifted(j) = col_min_shifted_rev(N-i_mid+1)(M-j) + c1*N
    i_dest(j) [the row achieving it] = N - src_rev(N-i_mid+1)(M-j)

  i.e. run the reversed sweep *through* row `k=N-i_mid` (inclusive, not
  "before" it as the forward checkpoint does), take its
  `col_min_shifted`/source snapshot from *after* that row, then reverse
  the column axis, shift the value by `+c1*N` and convert the row index
  via `N-k'`.

  `left_onehot_rev`/`right_cost_rev` must already be the *reversed*
  sequences (reverse frame order on both sides before building the
  cost-ready encoding - reversing the encoded arrays afterward would
  not reverse the underlying cost matrix correctly)."""
  del m  # kept in the signature for symmetry with ForwardCheckpoint's
  # (n_total, m_total) pair at the call site - not needed internally.
  k = n - i_mid
  f_rev, _, _, cms_after, src_after = _ChunkedSweep(left_onehot_rev,
                                                    right_cost_rev, c0, c1,
                                                    k, chunk)
  cost_from_mid = np.array(f_rev)[::-1]
  bcredit_shifted = np.array(cms_after)[::-1] + c1 * n
  i_dest = n - np.array(src_after)[::-1]
  return cost_from_mid, bcredit_shifted, i_dest


def StreamingBacktrack(left_onehot,
                       right_cost,
                       left_onehot_rev,
                       right_cost_rev,
                       c0,
                       c1,
                       n_total,
                       m_total,
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
  the current sub-range happens internally. `n_total`/`m_total` are the
  full sequence lengths (needed to correctly slice the reversed arrays
  for a sub-range)."""
  n_sub, m_sub = i_hi - i_lo, j_hi - j_lo
  if n_sub <= base_size or m_sub <= base_size:
    cost_sub = np.array(left_onehot[i_lo:i_hi] @ right_cost[j_lo:j_hi].T)
    return _BruteForceBacktrack(cost_sub, c0, c1, i_lo, j_lo)

  i_mid_local = n_sub // 2
  i_mid = i_lo + i_mid_local

  cost_to_mid, cms_before, src_before = ForwardCheckpoint(
      left_onehot[i_lo:i_hi], right_cost[j_lo:j_hi], c0, c1, i_mid_local,
      chunk)
  # Reversed sub-range of [i_lo,i_hi)/[j_lo,j_hi) within the full
  # reversed arrays.
  left_rev_sub = left_onehot_rev[n_total - i_hi:n_total - i_lo]
  right_rev_sub = right_cost_rev[m_total - j_hi:m_total - j_lo]
  cost_from_mid, bcredit, i_dest_full = BackwardCheckpoint(
      left_rev_sub, right_rev_sub, c0, c1, n_sub, m_sub, i_mid_local, chunk)
  # i_dest_full came back in [i_lo,i_hi)-local coordinates (n_sub was
  # passed as the local length) - shift to global, same as src_before.
  i_dest_full = i_dest_full + i_lo
  src_before_full = src_before + i_lo

  node_cost = cost_to_mid + cost_from_mid
  in_flight_cost = c0 + cms_before + bcredit
  crossing_cost = np.minimum(node_cost, in_flight_cost)
  j_star_local = int(np.argmin(crossing_cost))
  j_star = j_lo + j_star_local

  if node_cost[j_star_local] <= in_flight_cost[j_star_local]:
    SPLIT_STATS['node'] += 1
    left = StreamingBacktrack(left_onehot, right_cost, left_onehot_rev,
                              right_cost_rev, c0, c1, n_total, m_total,
                              i_lo, i_mid, j_lo, j_star, base_size, chunk)
    right = StreamingBacktrack(left_onehot, right_cost, left_onehot_rev,
                               right_cost_rev, c0, c1, n_total, m_total,
                               i_mid, i_hi, j_star, j_hi, base_size, chunk)
    return left + right

  SPLIT_STATS['in_flight'] += 1
  # The in-flight i-jump's true source/destination rows, recovered
  # directly from the checkpoints' source-row tracking (no full matrix
  # needed) - same reasoning as `_BacktrackRange`'s node_cost/
  # in_flight_cost comparison above, just without materializing the
  # full forward/backward matrices.
  i_source = int(src_before_full[j_star_local])
  i_dest = int(i_dest_full[j_star_local])

  left = StreamingBacktrack(left_onehot, right_cost, left_onehot_rev,
                            right_cost_rev, c0, c1, n_total, m_total, i_lo,
                            i_source, j_lo, j_star, base_size, chunk)
  jump_segment = [('ijump', i_source, j_star, i_dest, j_star)]
  right = StreamingBacktrack(left_onehot, right_cost, left_onehot_rev,
                             right_cost_rev, c0, c1, n_total, m_total,
                             i_dest, i_hi, j_star, j_hi, base_size, chunk)
  return left + jump_segment + right


# --- Public API ---------------------------------------------------------

_TERNARY_COST_TABLE = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]],
                               dtype=np.float32)
_TERNARY_EYE = np.eye(3, dtype=np.float32)


def OneHot(signatures: npt.ArrayLike) -> np.ndarray:
  """One-hot encodes a `(N, D)` array of ternary ({0,1,2}) per-frame
  descriptors (e.g. ffmpeg's MPEG-7 `signature` filter output) into a
  `(N, 3*D)` array such that `OneHot(left) @ CostTransform(right).T`
  gives the L1 distance matrix between every pair of frames as a single
  GEMM - what `StreamingBacktrack` needs as `left_onehot`/
  `left_onehot_rev`."""
  sig = np.asarray(signatures, dtype=np.int32)
  return _TERNARY_EYE[sig].reshape(sig.shape[0], -1)


def CostTransform(signatures: npt.ArrayLike) -> np.ndarray:
  """The `OneHot` GEMM's other operand - see `OneHot`'s docstring. What
  `StreamingBacktrack` needs as `right_cost`/`right_cost_rev`."""
  sig = np.asarray(signatures, dtype=np.int32)
  return _TERNARY_COST_TABLE[sig].reshape(sig.shape[0], -1)


@dataclasses.dataclass
class FrameDpParams:
  """Tunable cost-model parameters for `AlignFrames`.

  `c0`/`c1` define a jump's cost: skipping `k` frames on one side
  (content present in only one input) costs `c0 + c1*k`, independent of
  where the jump starts (see the module docstring). The defaults are
  the values measured against a real pair of differently-cut film
  releases (see `docs/frame_dp_align_notes.md`) - a reasonable starting
  point for similar material, not a universal constant; recalibrate
  (the same way `avtools`' `align_films.EstimateGapCost` measures
  `gap_cost` rather than guessing it) before relying on these for a
  very different kind of input.

  `base_size`/`chunk` are performance knobs for `StreamingBacktrack`,
  not part of the cost model: `base_size` is the sub-grid size below
  which Hirschberg's recursion switches to the small, full-backpointer
  base case; `chunk` is how many rows of the cost matrix are swept per
  GPU call in between.
  """
  c0: float = 2000.0
  c1: float = 280.0
  base_size: int = 500
  chunk: int = 80


@dataclasses.dataclass
class AlignedBlock:
  """One contiguous run of matched frames, `[left_start, left_end)` on
  the left side corresponding to `[right_start, right_end)` on the
  right (both ranges always the same length - a 1:1 frame match)."""
  left_start: int
  left_end: int
  right_start: int
  right_end: int


@dataclasses.dataclass
class UniqueBlock:
  """One contiguous run of frames present in only one of the two
  sequences - a jump, in `[start, end)` frame-index terms on that
  side."""
  side: str  # 'left' or 'right'.
  start: int
  end: int


@dataclasses.dataclass
class FrameAlignmentResult:
  aligned: List[AlignedBlock]
  unique: List[UniqueBlock]


def _PathToResult(path: List[Tuple[str, int, int, int, int]]
                  ) -> FrameAlignmentResult:
  """Collapses a raw `(move, i0, j0, i1, j1)` segment list (one match
  edge per segment, one segment per jump) into merged blocks - the
  shape of result `avtools`' `align_films.AlignmentResult` already
  uses, in frame-
  index rather than time terms."""
  aligned: List[AlignedBlock] = []
  unique: List[UniqueBlock] = []
  for move, i0, j0, i1, j1 in path:
    if move == 'diag':
      if (aligned and aligned[-1].left_end == i0 and
          aligned[-1].right_end == j0):
        aligned[-1].left_end = i1
        aligned[-1].right_end = j1
      else:
        aligned.append(AlignedBlock(i0, i1, j0, j1))
    elif move == 'ijump':
      unique.append(UniqueBlock('left', i0, i1))
    elif move == 'jjump':
      unique.append(UniqueBlock('right', j0, j1))
    else:
      raise ValueError(f'unknown move {move}')
  return FrameAlignmentResult(aligned=aligned, unique=unique)


def AlignFrames(left_signatures: npt.ArrayLike,
               right_signatures: npt.ArrayLike,
               params: FrameDpParams = FrameDpParams()) -> FrameAlignmentResult:
  """Aligns two per-frame descriptor sequences (e.g. MPEG-7 `signature`
  filter output, one row per frame, ternary {0,1,2} values) at frame
  resolution, returning which stretches matched and which are unique to
  one side - see `FrameAlignmentResult`.

  Runs the whole GPU streaming pipeline (`StreamingBacktrack`) under the
  hood; see the module docstring for the cost model and
  `docs/frame_dp_align_notes.md` for background. No confidence
  weighting or drift compensation is applied here - see
  `docs/frame_dp_align_future_work.md`.
  """
  left_sig = np.asarray(left_signatures)
  right_sig = np.asarray(right_signatures)
  n, m = left_sig.shape[0], right_sig.shape[0]

  left_onehot = OneHot(left_sig)
  right_cost = CostTransform(right_sig)
  left_onehot_rev = OneHot(left_sig[::-1])
  right_cost_rev = CostTransform(right_sig[::-1])

  path = StreamingBacktrack(left_onehot, right_cost, left_onehot_rev,
                            right_cost_rev, params.c0, params.c1, n, m, 0,
                            n, 0, m, params.base_size, params.chunk)
  return _PathToResult(path)
