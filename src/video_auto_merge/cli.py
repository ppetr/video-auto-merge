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
"""Command-line interface: `video-auto-merge align|merge LEFT RIGHT`."""
import argparse
import logging
import os
import pathlib
import sys
from typing import List, Optional

from video_auto_merge import align
from video_auto_merge import audio
from video_auto_merge import frame_dp_align
from video_auto_merge import merge
from video_auto_merge import plan
from video_auto_merge import report

_NO_HWACCEL = 'none'


def _DefaultCacheDir() -> pathlib.Path:
  base = os.environ.get('XDG_CACHE_HOME') or pathlib.Path.home() / '.cache'
  return pathlib.Path(base) / 'video-auto-merge' / 'signatures'


def _AddAlignmentArguments(parser: argparse.ArgumentParser) -> None:
  defaults = frame_dp_align.FrameDpParams()
  parser.add_argument('left', type=pathlib.Path, help='left video')
  parser.add_argument('right', type=pathlib.Path, help='right video')
  parser.add_argument(
      '--cache-dir',
      type=pathlib.Path,
      default=_DefaultCacheDir(),
      help='where per-frame signatures and alignments are cached '
      '(default: %(default)s)')
  parser.add_argument('--ffmpeg',
                      default='ffmpeg',
                      help='ffmpeg binary (default: %(default)s)')
  parser.add_argument(
      '--hwaccel',
      default='cuda',
      help=f'ffmpeg -hwaccel value for decoding, or "{_NO_HWACCEL}" '
      '(default: %(default)s)')
  parser.add_argument(
      '--c0',
      type=float,
      default=defaults.c0,
      help='flat cost of a jump over frames present in only one video '
      '(default: %(default)s)')
  parser.add_argument('--c1',
                      type=float,
                      default=defaults.c1,
                      help='per-frame cost of a jump (default: %(default)s)')


def _BuildParser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      prog='video-auto-merge',
      description='Automatic merging of two releases of the same film.')
  commands = parser.add_subparsers(dest='command', required=True)

  align_parser = commands.add_parser(
      'align',
      help='find what is shared by two videos and what is only in one',
      description='Aligns two videos at frame resolution and prints a table '
      'of the blocks in path order, i.e. the order in which they would '
      'appear in a film assembled from both. By default only blocks '
      'present in just one video are listed.')
  _AddAlignmentArguments(align_parser)
  align_parser.add_argument('--all',
                            action='store_true',
                            help='list matched blocks too')
  align_parser.add_argument('-o',
                            '--output',
                            type=pathlib.Path,
                            help='write the table to a file, not stdout')

  merge_parser = commands.add_parser(
      'merge',
      help='attach the right video\'s audio to the left video',
      description='Aligns the two videos, then writes the left video with '
      'all its streams unchanged plus the audio of the right video, cut '
      'to fit the left timeline: parts missing from the right video are '
      'filled from a fall-back audio stream of the left video, parts '
      'present only in the right video are dropped. Every cut is '
      'crossfaded.')
  _AddAlignmentArguments(merge_parser)
  merge_parser.add_argument('output', type=pathlib.Path, help='output file')
  merge_parser.add_argument(
      '--right-audio',
      type=int,
      action='append',
      metavar='INDEX',
      help='stream index of a right audio stream to attach (repeatable; '
      'default: all)')
  merge_parser.add_argument(
      '--fallback',
      type=int,
      metavar='INDEX',
      help='stream index of the left audio stream filling the gaps '
      '(default: its first audio stream)')
  merge_parser.add_argument(
      '--crossfade-ms',
      type=float,
      default=audio.DEFAULT_CROSSFADE_S * 1000,
      help='crossfade length at every cut (default: %(default)s)')
  merge_parser.add_argument(
      '--min-match',
      type=float,
      default=plan.DEFAULT_MIN_MATCH_S,
      help='matched blocks shorter than this many seconds are taken from '
      'the fall-back (default: %(default)s)')
  merge_parser.add_argument('--ffprobe',
                            default='ffprobe',
                            help='ffprobe binary (default: %(default)s)')
  return parser


def main(argv: Optional[List[str]] = None) -> int:
  """Runs the command-line tool; returns the process exit code."""
  args = _BuildParser().parse_args(argv)
  logging.basicConfig(level=logging.WARNING,
                      format='%(asctime)s %(message)s',
                      stream=sys.stderr)
  logging.getLogger('video_auto_merge').setLevel(logging.INFO)
  pre_input_args = ([] if args.hwaccel == _NO_HWACCEL else
                    ['-hwaccel', args.hwaccel])
  alignment = align.AlignVideos(args.left,
                                args.right,
                                args.cache_dir,
                                frame_dp_align.FrameDpParams(c0=args.c0,
                                                             c1=args.c1),
                                ffmpeg_path=args.ffmpeg,
                                pre_input_args=pre_input_args)
  if args.command == 'merge':
    merge.MergeAudio(args.left,
                     args.right,
                     alignment,
                     args.output,
                     right_streams=args.right_audio,
                     fallback_stream=args.fallback,
                     crossfade_s=args.crossfade_ms / 1000,
                     min_match_s=args.min_match,
                     ffmpeg_path=args.ffmpeg,
                     ffprobe_path=args.ffprobe)
    return 0
  table = report.FormatTable(report.OrderedBlocks(alignment.result),
                             alignment.left_times,
                             alignment.right_times,
                             only_differences=not args.all)
  if args.output:
    args.output.write_text(table)
  else:
    sys.stdout.write(table)
  return 0


if __name__ == '__main__':
  sys.exit(main())
