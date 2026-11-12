# Future work: frame-level DP alignment

Ideas deliberately not included in `src/video_auto_merge/frame_dp_align.py`, and why. See
`frame_dp_align_notes.md` for the approaches that *are* included and how
they were validated.

## A third edge for constant speed mismatches ("drift")

Old dub tracks are sometimes a frame-rate conversion (e.g. PAL/NTSC) of a
different master, which shows up as a slow constant drift rather than a
discrete insertion/removal. A third edge type - a single-frame "drift
step," cheaper than a jump but more expensive than a confident match -
was prototyped to absorb that kind of drift without the model confusing
it for a real cut.

Needs `c1 < eps < c0+c1` (`eps` the drift step's cost) to behave
sensibly: below `c1`, a run of drift steps becomes a strictly cheaper
way to explain *any* jump, including real multi-frame edits; above
`c0+c1`, a single drift step is never preferred even for a lone
one-frame correction, making it pointless.

Adds real complexity beyond the shipped two-edge model: the per-row
recurrence needs a second, coupled prefix-min scan instead of one (the
drift edge creates a direct dependency on the row's own previous column,
not just on earlier rows), and the Hirschberg correction needs a
separate forward/backward "credit" channel in addition to the one the
jump edge already needs. Implemented and validated against a
brute-force reference in isolation, but no case has yet needed it in
practice - calibrating it to be effectively inert produced the same
result as leaving it out. Candidate for revisiting if a real case of
genuine constant-rate drift (as opposed to discrete insertions) turns
up.

## "Cooldown" refinement to the drift idea

A variant raised but not designed or implemented: let a drift step's
cost increase the more recently it's been used (discouraging a long run
of drift steps from masquerading as an acceptable substitute for a real
jump), decaying back down after a run of genuine matches. Would need its
own state-augmented DP (cost depends on recent history, not just
position), not just a parameter tweak to the plain drift edge above.

## Soft-min / log-sum-exp cost combination

A temperature-parameterized smooth approximation to the hard minimum
used throughout this module, applied to the same edge set. Implemented
and partially validated, but with two open problems: the within-row
prefix-min trick that makes `FastFrameDp` O(N*M) does not carry over -
soft-min's within-row recursion needs a genuinely sequential per-column
scan, with no known reduction to a parallel GPU-friendly form yet - and
no concrete benefit over hard-min was established for this use case
either way. Not pursued further unless a specific need for a smooth
(rather than sharp) cost combination comes up.

## `FrameConfidence` weighting

The MPEG-7 `signature` filter reports a per-frame confidence value
alongside the descriptor itself. The earlier 1-second-block banded
alignment weighed match cost by
`min(confidence_left, confidence_right)` so that low-information
frames (static/dark scenes, titles) don't force a
falsely-confident-looking decision. `frame_dp_align` does not do this
yet - plausibly still a good idea, but not validated at frame
resolution.

## Recalibrating `c0`/`c1`

The current defaults (see `FrameDpParams`) are one measured data point,
not derived from first principles. Now that `Backtrack`/
`StreamingBacktrack` expose the true optimal path rather than an
approximate or local diagnostic, comparing the resulting path's cost
(and qualitative behavior - does it jump where it obviously shouldn't,
or fail to jump where it obviously should) across several candidate
`c0`/`c1` values is straightforward in principle, and should probably be
done - with a proper measurement procedure, not a guess - before relying
on the current defaults for anything beyond the material they were
measured on. The earlier 1-second-block model measured its gap cost from
each film pair's own data rather than using a fixed constant; nothing
equivalent exists yet for this frame-level model.

## Whole-film-scale independent cost verification

`PathCost` recomputes a path's cost directly from its segments as a
sanity check independent of the DP machinery that produced it - but it
needs the full cost matrix in memory, so it only works at the scale
`NaiveFrameDp`/`Backtrack`'s tests use. A streaming equivalent, able to
verify a whole-film-scale path's cost without materializing the full
cost matrix (summing each segment's contribution from chunked GEMMs, the
same way `StreamingBacktrack` itself avoids materializing it), does not
exist yet - would be useful as an independent check on a real run,
separate from trusting the DP/backtracking code path that produced the
path in the first place.

## Whole-film-scale GPU memory regression test

The three GPU memory-management pitfalls documented in
`frame_dp_align_notes.md` (JIT-recompilation leak, device-array-slicing
accumulation during recursion, allocator configuration) were each
confirmed only by running the real pipeline at whole-film scale - the
unit tests' tiny inputs (a few dozen frames per side) exercise the same
code paths for *correctness*, but not at a scale where any of those
three bugs would reproduce. There is currently no automated test that
would catch a regression of any of them (e.g. someone reintroducing a
closure-based jit target "for clarity"). Adding one would mean
generating or fetching a large-enough synthetic signature pair (likely
tens of thousands of frames per side - real two-hour-film scale) and
running `StreamingBacktrack` against it under a memory budget, probably
as a separate `@pytest.mark.slow`/manual script rather than part of the
default `pytest test/` run.
