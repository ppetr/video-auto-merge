# Merging: design and open items

How an alignment turns into an output file. Implemented in
`src/video_auto_merge/plan.py` (alignment to segment plan) and the audio
assembly built on it.

## Model

- The **left** film is the base. Its video and all its streams are copied
  unchanged; the output timeline is the left timeline, so the output has
  exactly the left film's length and A/V sync cannot drift.
- Streams of the **right** film (audio, text subtitles) are attached to it.
  For each part of the left timeline:
  - matched block: take the right stream, shifted onto the left time;
  - block only in the left film (the right one has a gap): fill from a
    *fall-back* left audio track, converted to the right track's sample
    rate and channel layout;
  - block only in the right film: dropped.
- Every audio join gets a short symmetric crossfade (default 20 ms,
  equal-power quarter-sine curve). Each segment is extended by half the
  crossfade past each join, so the output length stays sample-exact.
- To put a dub on the longer of two releases, swap left and right.
- Timestamps always come from the per-frame signatures (frame PTS), never
  from `index * frame_period`. Times are on ffmpeg's timeline (container
  start = 0); audio streams are decoded with their start padded to that
  timeline, since streams usually start a few ms after the container.
- Default output codec: Opus (VBR, about 64 kbit/s per channel); text
  subtitles as SRT.

## Open items

- Cut precision is one video frame (about 42 ms): the true audio edit in
  the right film may lie elsewhere by up to about half a frame, i.e. roughly
  the crossfade length. Refining cuts by audio cross-correlation is a
  possible later step.
- Matched blocks shorter than a threshold (default 0.5 s, to be tuned on the
  real table) are replaced by the fall-back, because they cannot be
  crossfaded reasonably between two jumps.
- No loudness matching between the right track and the fall-back; a level
  jump at a join is audible if the masters differ. Possible later option.
- One global fall-back track; a per-output-track choice only if needed.
- Image-based subtitles (PGS, VobSub) cannot be re-timed and are skipped;
  text subtitles are re-timed per block (cues in dropped blocks are
  discarded, cues crossing a cut are clipped).
- Copying the right audio without re-encoding (when the alignment is one
  block plus trimmed ends) is not implemented; everything is re-encoded.
- Subtitles that fill gaps without any audio are not implemented.
- Constant-speed drift and trimming of fade-to-black ends are deferred, see
  [frame_dp_align_future_work.md](frame_dp_align_future_work.md).
- `libopus` rejects some channel layouts (e.g. `5.1(side)`), which then
  need an explicit `channelmap`.
