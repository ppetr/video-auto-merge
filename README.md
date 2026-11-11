# video-auto-merge

_*Disclaimer:* This is not an officially supported Google product._

## What this is

A GPU-accelerated library for aligning two per-frame video descriptor
sequences (e.g. ffmpeg's MPEG-7 `signature` filter output) at frame
resolution - the goal being to splice two releases of the same film
together, such as attaching an old dub track onto a newer or
re-cut/extended video release, even when the two releases differ by
scattered insertions and removals rather than a single constant offset.

Split out of `avtools` into its own project: the
GPU/JAX dependency and the dynamic-programming/divide-and-conquer
algorithm it relies on are both substantial and technologically
unrelated to the rest of that toolkit's ffmpeg-wrapper scripts, so this
grew into a project of its own rather than one more tool among many.

See `docs/frame_dp_align_notes.md` for the design history (what was
tried, what worked) and `docs/frame_dp_align_future_work.md` for ideas
deliberately deferred.

## Status

Early stage. The core algorithm (`src/frame_dp_align.py`) is validated
against brute-force references on synthetic data and has been run
successfully end-to-end on a real pair of differently-cut film
releases, but there is no command-line tool yet - `AlignFrames` is a
library function, not a script.

## Requirements

- `jax` (CPU-only works for correctness; a CUDA-enabled `jaxlib` is
  needed for whole-film-scale performance - see `frame_dp_align.py`'s
  module docstring for the cost model and `FrameDpParams` for the
  tunable parameters).

## Contributions

Please see [Code of Conduct](docs/code-of-conduct.md) and
[Contributing](docs/contributing.md).
