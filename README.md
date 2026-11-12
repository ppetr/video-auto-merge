# video-auto-merge

_*Disclaimer:* This is not an officially supported Google product._

## What this is

A tool that takes two releases of the same film and produces a single
merged result with ffmpeg - for example, attaching an old dub track onto
a newer or re-cut/extended video release, even when the two releases
differ by scattered insertions and removals rather than a single constant
offset. Where the second release has content with no matching audio,
subtitles may be added instead.

The intended pipeline, end to end:

1. Compute per-frame descriptors of both videos (ffmpeg's MPEG-7
   `signature` filter).
2. Align the two descriptor sequences at frame resolution (GPU-accelerated
   dynamic programming, no band or pre-estimated offset required).
3. Turn the alignment into a list of matched and unmatched segments.
4. Cut and join the audio/video segments with ffmpeg.
5. Optionally fill the gaps that have no audio with subtitles.
6. Write out the complete result.

Only steps 1 and 2 are implemented so far - see [Status](#status).

Split out of [avtools](https://github.com/ppetr/avtools) into its own
project: the GPU/JAX dependency and the dynamic-programming/divide-and-conquer
algorithm it relies on are both substantial and technologically
unrelated to the rest of that toolkit's ffmpeg-wrapper scripts, so this
grew into a project of its own rather than one more tool among many.

See [docs/frame_dp_align_notes.md](docs/frame_dp_align_notes.md) for the
design history (what was tried, what worked) and
[docs/frame_dp_align_future_work.md](docs/frame_dp_align_future_work.md)
for ideas deliberately deferred.

## Status

Early stage (pre-alpha). Only the alignment core exists; there is no
command-line tool yet - `AlignFrames` is a library function, not a script.

| Component | State |
|---|---|
| Frame-level DP alignment ([`AlignFrames`](src/video_auto_merge/frame_dp_align.py)) | Implemented; validated against brute-force references on synthetic data |
| GPU/streaming path for whole-film-length inputs | Implemented; correctness tests only, no large-scale stress test |
| Descriptor extraction (ffmpeg `signature` filter, caching) | Implemented ([`signatures.py`](src/video_auto_merge/signatures.py)); library only |
| Alignment path to segment list (matched / unmatched spans, timestamps) | Not implemented |
| Cutting and joining audio/video with ffmpeg | Not implemented |
| Subtitles for spans without matching audio | Not implemented (planned, optional) |
| Constant-speed drift compensation (e.g. PAL/NTSC) | Not implemented (deferred, see [future work](docs/frame_dp_align_future_work.md)) |
| Command-line interface | `video-auto-merge align` only (see [Usage](#usage)) |

The alignment algorithm itself was validated on a real pair of
differently-cut film releases in a prototype of this code; it has not
yet been re-run against real data in this repository.

## Usage

```
video-auto-merge align film-L.mkv film-R.mkv --cache-dir signatures/
```

Prints a table of the blocks present in only one of the two videos, in the
order in which they would appear in a film assembled from both (`out` is
the block's start in that film). `--all` lists matched blocks too, `-o`
writes the table to a file. Per-frame signatures are slow to compute (they
use ffmpeg, with `-hwaccel cuda` decoding by default; `--hwaccel none`
turns it off), so they are cached in `--cache-dir` and reused on later
runs. The alignment itself needs a GPU for whole-film inputs (see
Requirements); `--c0`/`--c1` set the cost of a jump over frames present
in only one video.

## Requirements

- `ffmpeg` in `$PATH`, for the per-frame signatures.
- `jax` (CPU-only works for correctness; a CUDA-enabled `jaxlib` is
  needed for whole-film-scale performance - see the module docstring of
  [`src/video_auto_merge/frame_dp_align.py`](src/video_auto_merge/frame_dp_align.py) for the cost model and `FrameDpParams` for the
  tunable parameters).

## Testing

```
pip install -e .[test]
pytest test/ -v
```

`jax` picks whichever backend its installed `jaxlib` provides - there
is no test-specific flag for this. Plain `pip install -e .[test]`
pulls in a CPU-only `jaxlib` by default, so the suite above runs
(correctly, just slower) on CPU.

To actually exercise the GPU path, install a CUDA-enabled `jaxlib`
instead (matching your CUDA version, e.g. `pip install -U
"jax[cuda12]"`), then explicitly run the `gpu`-marked suite in
[`test/test_frame_dp_align_gpu.py`](test/test_frame_dp_align_gpu.py) - the *same* tests as above, just
re-collected under a fixture that fails loudly if jax doesn't resolve
to a real GPU device, rather than silently running on CPU under a
false "GPU-tested" label:

```
pytest -m gpu -v
```

This is excluded from a plain `pytest`/`pytest test/` run on purpose
(see `addopts` in [`pyproject.toml`](pyproject.toml)) - a routine test run shouldn't
require a GPU to pass. `pytest test/test_frame_dp_align_gpu.py` alone
(without `-m gpu`) will report 0 selected for the same reason; the
marker has to be given explicitly.

Note that both suites use tiny inputs (a few dozen frames per side),
so they check *correctness* of the GPU code path, not the whole-film-
scale memory behavior documented in
[docs/frame_dp_align_notes.md](docs/frame_dp_align_notes.md) (the three GPU memory pitfalls found during development only manifest
around ~10^5 x 10^5-scale inputs). There is no large-scale stress test
in this repo yet - see
[docs/frame_dp_align_future_work.md](docs/frame_dp_align_future_work.md).

## Contributions

Please see [Code of Conduct](docs/code-of-conduct.md) and
[Contributing](docs/contributing.md).
