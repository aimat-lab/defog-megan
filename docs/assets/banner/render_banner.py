#!/usr/bin/env python3
"""Render the README banner: every frame through figlib / VectorCraft, then the GIF through ffmpeg.

Needs the ``figure`` skill (its figlib drawing library and build.py, found through $FIGURE_SKILL, default
~/.claude/skills/figure), VectorCraft's command-line program, Pillow and ffmpeg. These are the author's figure
tools, so the GIF and trajectory.json are the artefacts a reader can rely on; this is how they were made.

    python3 render_banner.py frames --work WORK [--scale 3.42] [--only 0,50,100] [--jobs 6]
    python3 render_banner.py strip  --work WORK --out keyframes.png [--only 8,40,...]
    python3 render_banner.py gif    --work WORK --out banner.gif [--fps 15] [--width 1600]

``frames`` writes WORK/png/fNNN.png (white background); ``strip`` lays selected frames out on one sheet for a
look; ``gif`` assembles all frames with one shared palette. The data comes from $BANNER_DATA (default
trajectory.json next to this file).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import re
import shutil
import subprocess
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.expanduser(os.environ.get("FIGURE_SKILL", "~/.claude/skills/figure"))
BUILD = os.path.join(SKILL, "scripts", "build.py")
DEFAULT_SCALE = 1600 / 468          # px per pt for a 1600 px wide banner


def banner_module():
    sys.path.insert(0, os.path.join(SKILL, "lib"))
    sys.path.insert(0, HERE)
    import banner
    return banner


def frame_png(work, k):
    return os.path.join(work, "png", f"f{k:03d}.png")


def render_frame(work, k, scale):
    """Render frame k; returns the layout warnings the build printed (empty when clean)."""
    tmp = os.path.join(work, "tmp", f"{k:03d}")
    os.makedirs(tmp, exist_ok=True)
    env = dict(os.environ, BANNER_FRAME=str(k))
    proc = subprocess.run([sys.executable, BUILD, os.path.join(HERE, "banner.py"), "--work", tmp,
                           "--scale", f"{scale:.5f}"], env=env, capture_output=True, text=True, timeout=300)
    src = os.path.join(tmp, "banner.png")
    if proc.returncode or not os.path.exists(src):
        raise RuntimeError(f"frame {k} failed:\n{proc.stdout[-800:]}\n{proc.stderr[-800:]}")
    os.makedirs(os.path.dirname(frame_png(work, k)), exist_ok=True)
    shutil.move(src, frame_png(work, k))
    shutil.rmtree(tmp, ignore_errors=True)
    return [ln for ln in proc.stdout.splitlines() if ln.startswith("WARNING")]


def parse_only(text, total):
    if not text:
        return list(range(total))
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def cmd_frames(args):
    total = banner_module().FRAMES
    todo = parse_only(args.only, total)
    warned = {}
    with cf.ThreadPoolExecutor(args.jobs) as pool:
        futs = {pool.submit(render_frame, args.work, k, args.scale): k for k in todo}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            k = futs[fut]
            w = fut.result()
            if w:
                warned[k] = w
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} frames", flush=True)
    for k in sorted(warned):
        for line in warned[k]:
            print(f"  frame {k}: {line}")
    print(f"{len(todo)} frames in {os.path.join(args.work, 'png')} ({len(warned)} with layout warnings)")


def cmd_strip(args):
    total = banner_module().FRAMES
    ks = parse_only(args.only, total)
    missing = [k for k in ks if not os.path.exists(frame_png(args.work, k))]
    if missing:
        with cf.ThreadPoolExecutor(args.jobs) as pool:
            list(pool.map(lambda k: render_frame(args.work, k, args.scale), missing))
    ims = [Image.open(frame_png(args.work, k)).convert("RGB") for k in ks]
    w = int(ims[0].width * args.shrink)
    h = int(ims[0].height * args.shrink)
    cols = args.cols
    rows = (len(ims) + cols - 1) // cols
    gap = 6
    sheet = Image.new("RGB", (cols * w + (cols + 1) * gap, rows * h + (rows + 1) * gap), (205, 205, 205))
    for i, im in enumerate(ims):
        r, c = divmod(i, cols)
        sheet.paste(im.resize((w, h), Image.LANCZOS), (gap + c * (w + gap), gap + r * (h + gap)))
    sheet.save(args.out)
    print(f"wrote {args.out}: frames {ks} ({sheet.width}x{sheet.height})")


def cmd_gif(args):
    total = banner_module().FRAMES
    last = total - 1
    a, b = frame_png(args.work, 0), frame_png(args.work, last)
    if open(a, "rb").read() == open(b, "rb").read():
        total -= 1                     # the last frame IS the first: drop it so that the loop has no double frame
        print("last frame equals the first: dropped it for a seamless loop")
    pal = os.path.join(args.work, "palette.png")
    inp = ["-framerate", str(args.fps), "-start_number", "0", "-i", os.path.join(args.work, "png", "f%03d.png")]
    scale = f"scale={args.width}:-1:flags=lanczos," if args.width else ""
    nframes = ["-frames:v", str(total)]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inp, *nframes, "-vf",
                    f"{scale}palettegen=stats_mode=full:max_colors={args.colors}", pal], check=True)
    use = f"paletteuse=dither={args.dither}" + (":bayer_scale=%d" % args.bayer if args.dither == "bayer" else "") \
          + ":diff_mode=rectangle"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inp, "-i", pal, *nframes, "-lavfi",
                    f"{scale[:-1]} [x]; [x][1:v] {use}" if scale else f"[0:v][1:v] {use}", "-loop", "0", args.out],
                   check=True)
    size = os.path.getsize(args.out)
    print(f"wrote {args.out}: {total} frames at {args.fps} fps = {total / args.fps:.1f} s, {size / 1e6:.2f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("frames", cmd_frames), ("strip", cmd_strip), ("gif", cmd_gif)):
        p = sub.add_parser(name)
        p.add_argument("--work", required=True)
        p.add_argument("--scale", type=float, default=DEFAULT_SCALE, help="px per pt of the frames")
        p.add_argument("--jobs", type=int, default=6)
        p.add_argument("--only", help="frames, e.g. 0,50,100-120 (default: all)")
        p.set_defaults(fn=fn)
        if name == "strip":
            p.add_argument("--out", required=True)
            p.add_argument("--cols", type=int, default=2)
            p.add_argument("--shrink", type=float, default=0.5)
        if name == "gif":
            p.add_argument("--out", required=True)
            p.add_argument("--fps", type=int, default=15)
            p.add_argument("--width", type=int, default=0, help="output width in px (default: as rendered)")
            p.add_argument("--colors", type=int, default=256)
            p.add_argument("--dither", default="sierra2_4a", choices=["bayer", "sierra2_4a", "none"])
            p.add_argument("--bayer", type=int, default=4)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
