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

## Testing

```
pip install -e .[test]
pytest test/ -v
```

`jax` picks whichever backend its installed `jaxlib` provides - there
is no test-specific flag for this. Plain `pip install -e .[test]`
pulls in a CPU-only `jaxlib` by default, so the suite above runs
(correctly, just slower) on CPU. To actually exercise the GPU path,
install a CUDA-enabled `jaxlib` instead (matching your CUDA version,
e.g. `pip install -U "jax[cuda12]"`) and confirm it *before* running
tests:

```
python -c "import jax; print(jax.devices())"
```

A `CudaDevice` in that output (not `CpuDevice`) means the same test
command now runs its JAX code on the GPU - nothing else changes.

Note that the unit tests use tiny inputs (a few dozen frames per side),
so they check *correctness* of the GPU code path, not the whole-film-
scale memory behavior documented in `docs/frame_dp_align_notes.md`
(the three GPU memory pitfalls found during development only manifest
around ~10^5 x 10^5-scale inputs). There is no large-scale stress test
in this repo yet - see `docs/frame_dp_align_future_work.md`.

## Contributions

Please see [Code of Conduct](docs/code-of-conduct.md) and
[Contributing](docs/contributing.md).
