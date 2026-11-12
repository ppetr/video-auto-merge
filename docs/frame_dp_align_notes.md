# Design notes: frame-level DP alignment

Background for `src/video_auto_merge/frame_dp_align.py`, a frame-resolution alternative to
an earlier, simpler 1-second-block banded alignment. This document
summarizes, in general terms, what was tried while developing it and what
the outcome was - so a future reader knows which direction already turned
out to work, without having to re-derive it. Deliberately deferred ideas
live in `frame_dp_align_future_work.md` instead of here.

## The problem, restated

The earlier banded alignment aligned two releases of the same film at
1-second block resolution, inside a band around a pre-estimated offset,
with a gap penalty for inserted/removed content and a minimum-run-length
heuristic to reject spurious short "matches" inside a real gap. It works,
but has two structural weak points: the band width is a heuristic guess,
and 1-second resolution means every matched span still needs a separate
fine-alignment pass to reach frame accuracy.

This explores whether going to full frame resolution, with no band at
all, removes both problems at once - and whether it can still be made to
run in reasonable time and memory on whole-film-length inputs.

## Approaches tried

### 1. Unrestricted frame-level alignment with a flat jump cost

Model: match adjacent frames at a cost equal to their descriptor distance,
or "jump" past any number of consecutive frames on one side at a cost
that is flat per jump (`c0`) plus linear in how many frames it skips
(`c1` per frame) - deliberately *not* scaled by where the jump starts, so
the model doesn't care whether a difference falls on a scene cut or in
the middle of a continuous shot. No band restriction: every possible
jump start/end is implicitly considered.

This removes the band-width heuristic entirely (the DP searches
everything, cheaply, by construction - see the prefix-min point below)
and, run at per-frame resolution, removes the need for a separate
fine-alignment pass too.

### 2. Hard-min vs. soft-min cost combination

Two ways to combine competing candidate costs at each cell were
prototyped: the ordinary hard minimum, and a temperature-parameterized
soft minimum (log-sum-exp based), which gives a smooth approximation
instead of a sharp choice.

Outcome: the soft-min variant was implemented and partially validated,
but deprioritized. It turned out harder to calibrate and reason about
(an extra temperature parameter, awkward to pick) without a demonstrated
benefit over the simpler hard-min model for this use case. The shipped
library uses hard-min only.

### 3. A third "drift" edge for constant speed mismatches

Old dub tracks are sometimes a PAL/NTSC frame-rate conversion of a
different master, which shows up as a slow constant drift rather than a
discrete insertion/removal. A third edge type was prototyped: a
single-frame "drift step" (advance one frame on one side only, at a
fixed per-frame cost between the match cost and the jump cost) meant to
absorb that kind of drift cheaply without it being confused for a real
cut.

Outcome: implemented and validated in isolation, but not included in the
shipped model (see future-work doc). It adds real complexity - the
per-row recurrence needs a second coupled prefix-min scan instead of one,
and the linear-memory backtracking needs extra forward/backward
bookkeeping - and no case needing it has been confirmed yet; calibrating
it to be effectively inert produced the same result as leaving it out
entirely. Keeping the shipped recurrence to two edge types (match, jump)
keeps both the per-row recurrence and the backtracking correction
noticeably simpler.

### 4. The all-pairs-jump recurrence, made tractable

Naively, "jump from anywhere before the current position" requires
scanning every possible jump origin for every cell - cubic-ish in the
input size, intractable at whole-film scale. This reduces to a
prefix-minimum of a shifted version of the per-cell candidate costs,
computed once per row/column sweep rather than re-scanned per cell -
turning an expensive explicit search into a linear scan. Validated
against a brute-force, explicit-scan reference on small synthetic inputs
before being trusted at any larger scale; this is what makes whole-film,
unrestricted-jump frame-level alignment computationally feasible at all.

### 5. Linear-memory backtracking (Hirschberg's divide-and-conquer)

The full cost matrix doesn't fit in memory at whole-film scale, so the
actual alignment path (not just its cost) needs a divide-and-conquer
backtracking scheme that only ever holds a couple of rows/columns at
once.

The textbook version of this needed one real correction, not just a
direct application: because a jump's cost is flat rather than scaling
with distance, a single jump can start before the recursion's current
split point and land after it without the optimal path ever having a
node exactly on the split - so naively forcing the path through the
split row would double-charge the jump's fixed cost. Fix: track, at the
split, not just the ordinary "best path through this exact row" value
but also a separate "jump in flight" value, derived from the same
running prefix-minimum the forward/backward sweeps already compute. This
was validated by itself (does it recover the right crossing point and
jump endpoints?) and then end-to-end (does the full reconstructed path's
cost match the known optimum?) on small synthetic inputs, both with
hard-coded/explicit-loop reference implementations, before moving to any
optimized version.

### 6. GPU port, and three memory-management pitfalls worth remembering

The prefix-min sweep and the Hirschberg correction both port
straightforwardly to batched GPU array operations - but getting there
(on JAX) surfaced three separate memory bugs that are worth remembering
for any similar batched-GPU-DP work, since none of them follow from a
straightforward estimate of the data volume involved:

- Defining the compiled per-chunk computation as a closure created fresh
  on every call, instead of once at module scope, meant every call
  compiled (and kept alive) a brand new executable. A divide-and-conquer
  recursion that calls back into this hundreds of times accumulated
  enough distinct compiled executables to exhaust GPU memory well before
  any plausible estimate of the actual data size would predict.
- Keeping the (large) input encodings as on-device arrays and slicing
  them per recursive call, instead of keeping them as plain host-side
  arrays and only moving the current slice to the device, accumulated a
  separate device-side copy per outstanding call-stack frame during the
  recursion - again, a memory bug tied to *how* the recursion holds
  references, not to the data volume itself.
- A GPU allocator configuration that disables pre-allocation (seemingly
  the conservative, memory-safe choice for a large job) made
  fragmentation-driven allocation failures *more* likely, not less; the
  default pre-allocating allocator worked correctly. Counter-intuitive
  enough to be worth writing down.

All three were confirmed as the actual cause of out-of-memory failures
(not merely suspected) before being fixed, and the fixes are baked into
the shipped implementation rather than left as something a future caller
needs to rediscover.

### 7. End-to-end validation against the true reconstructed path

Once the GPU-accelerated backtracking was validated against a CPU
reference on small synthetic inputs, it was run on a real pair of
differently-cut releases of the same film, and the *actual* reconstructed
optimal path was inspected directly at a location with a known
difference between the two releases - not an intermediate proxy like
"which column has the lowest cost in this one row," which earlier
exploration had already shown can be misleading (see "what didn't work"
below).

Result: the path crossed via a single jump, located almost exactly where
independently expected, with clean, frame-by-frame matches immediately
on both sides of it and no other jumps nearby. This was the first time
the model's actual global optimum (rather than a partial diagnostic) was
inspected directly, and it confirmed the whole chain - cost model,
calibration, and GPU implementation - end-to-end on real data, not just
on synthetic test cases.

## What didn't work / was abandoned

- **Alignment at the granularity of detected scene cuts.** A real
  difference between two releases is often *inside* a continuous shot
  (a mid-scene trim), not at a cut boundary - the shorter release simply
  has no detectable cut where the content was removed. Scene-level units
  are too coarse to represent that case at all.
- **"Longest run of low match-distance = a genuine match" as a boundary
  detector.** A short run of coincidentally-similar content can appear
  *inside* a real difference, not just outside it, producing a
  false-positive "resync" in the middle of a genuine gap. This is exactly
  the failure mode the flat jump cost (rather than any kind of
  minimum-run-length threshold) is meant to make non-viable: a short
  island of matches surrounded by a gap on both sides costs two jumps
  instead of one, so it's only chosen when it is genuinely cheaper, not
  merely plausible-looking.
- **Trusting an intermediate "best match in this row/column" diagnostic
  as a stand-in for the true global path.** Confirmed misleading in
  earlier exploration (an apparent answer that didn't survive full
  backtracking) - this is the main reason direct backtracking was
  prioritized at all, rather than continuing to lean on cheaper proxy
  diagnostics.

## Parameters

`c0` (flat per-jump cost) and `c1` (per-frame jump cost) are measured,
not derived from first principles - see `FrameDpParams` in
`frame_dp_align.py` for the current defaults and how they were obtained.
General guidance for recalibrating on different material: `c1` needs to
be low enough that a real multi-frame difference is clearly cheaper to
represent as one jump than as many short ones, but high enough that the
model doesn't prefer jumping over a handful of merely-expensive-to-match
frames instead of actually matching them.

Real-data sweep of `c0` (one pair of differently-cut releases, c1 = 280):
with `c0` = 2000 the path contained many 1-5 frame jumps that look like
noise rather than real edits (21 on the first ~15 minutes, 65 over the
whole film on the left/right side combined); `c0` = 8000 cut these to 6
and 31 respectively while the long-range structure (and the known
649-frame insert) stayed identical. `c0` = 16000 started to produce a
false diagonal match at the start of the film, so 8000 is the default.
