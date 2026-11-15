"""Render an 18-second walkthrough from actual CLI JSON output.

Media-only dependencies: pillow, imageio, imageio-ffmpeg, numpy.
Run from the checkout: python scripts/render_demo.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
reports = {}
for name, expected in (("pass", 0), ("drift", 1)):
    args = [
        "-m",
        "llm_drift_bench",
        "run",
        "-b",
        "examples/baseline.jsonl",
        "-c",
        f"examples/candidate-{name}.jsonl",
        "-p",
        "examples/extraction-suite.json",
        "--format",
        "json",
    ]
    result = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env=ENV,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == expected, result.stderr
    reports[name] = {
        "command": "python " + " ".join(args),
        "exit_code": result.returncode,
        "report": json.loads(result.stdout),
    }
(OUT / "demo-provenance.json").write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    paths = ["C:/Windows/Fonts/consolab.ttf"] if bold else ["C:/Windows/Fonts/consola.ttf"]
    paths += ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise RuntimeError("Install Consolas or DejaVu Sans Mono to render the demo.")


TITLE, BODY, SMALL = font(44, True), font(26), font(20)
BG, PANEL, WHITE, MUTED = "#080e1c", "#101d30", "#eaf2ff", "#91a8c5"
CYAN, GREEN, RED = "#63d8ef", "#67e5ab", "#ff8596"
W, H, FPS, SECONDS = 1280, 720, 20, 18


def frame(t: float) -> Image.Image:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((64, 30), "llm-drift-bench", font=TITLE, fill=WHITE)
    d.text((65, 91), "Catch regressions before your next model change.", font=BODY, fill=MUTED)
    d.rounded_rectangle((64, 152, 1216, 628), radius=20, fill=PANEL, outline="#253b56", width=2)
    for x, color in ((92, RED), (116, "#f3cf75"), (140, GREEN)):
        d.ellipse((x, 173, x + 12, 185), fill=color)
    d.text((172, 167), "OFFLINE DEMO  /  0.2.0 development", font=SMALL, fill=MUTED)
    d.line((64, 204, 1216, 204), fill="#253b56", width=2)
    if t < 4:
        label, color = "01  /  START WITH SAVED OUTPUTS", CYAN
        lines = [
            "Same prompts. Two captured response files.",
            "",
            'baseline:   {"total":100}',
            'candidate:  {"total":180}',
            "",
            "Will your application still get the fields it expects?",
        ]
    elif t < 9:
        label, color = "02  /  CHECK AN UNCHANGED CANDIDATE", GREEN
        p = reports["pass"]["report"]
        lines = [
            "$ python -m llm_drift_bench run",
            "    -b examples/baseline.jsonl",
            "    -c examples/candidate-pass.jsonl",
            "    -p examples/extraction-suite.json",
            "",
            f"PASS   |   {p['total_samples']} samples   |   "
            f"drift {p['mean_composite_drift']:.4f}   |   exit 0",
        ]
    elif t < 15:
        label, color = "03  /  CATCH THE REGRESSION", RED
        p = reports["drift"]["report"]
        lines = [
            "$ python -m llm_drift_bench run",
            "    -b examples/baseline.jsonl",
            "    -c examples/candidate-drift.jsonl",
            "    -p examples/extraction-suite.json",
            "",
            f"FAIL   |   {p['drifted_samples']} drifted   |   "
            f"JSON conformity {p['candidate_json_conformity_rate']:.0%}",
            f"Mean drift {p['mean_composite_drift']:.4f}   |   exit 1",
        ]
    else:
        label, color = "04  /  MAKE MODEL CHANGES REVIEWABLE", CYAN
        lines = [
            "Changed invoice total. Missing required priority.",
            "",
            "Compare captures  >  inspect failures  >  gate CI",
            "",
            "Try the examples in the repository.",
            "github.com/anant-goel/llm-drift-bench",
        ]
    d.text((94, 230), label, font=BODY, fill=color)
    elapsed = t - (0 if t < 4 else 4 if t < 9 else 9 if t < 15 else 15)
    for i, line in enumerate(lines):
        if elapsed >= i * 0.1:
            d.text(
                (94, 285 + i * 41),
                line,
                font=BODY,
                fill=color if line.startswith(("PASS", "FAIL")) else WHITE,
            )
    d.text(
        (65, 650),
        "Handwritten fixtures. Actual CLI results. Edited walkthrough, not live timing.",
        font=SMALL,
        fill=MUTED,
    )
    d.rounded_rectangle((64, 692, 1216, 697), radius=2, fill="#253b56")
    d.rounded_rectangle(
        (64, 692, 64 + int(1152 * (t + 1 / FPS) / SECONDS), 697), radius=2, fill=CYAN
    )
    return im


gif_frames = []
with imageio.get_writer(
    str(OUT / "demo.mp4"),
    fps=FPS,
    codec="libx264",
    quality=8,
    macro_block_size=1,
    ffmpeg_params=["-movflags", "+faststart"],
) as writer:
    for n in range(FPS * SECONDS):
        im = frame(n / FPS)
        writer.append_data(np.asarray(im))
        if n % 4 == 0:
            gif_frames.append(im.resize((960, 540)).quantize(colors=96))
gif_frames[0].save(
    OUT / "demo.gif",
    save_all=True,
    append_images=gif_frames[1:],
    duration=200,
    loop=0,
    optimize=True,
)
frame(12).save(OUT / "demo-poster.png")
print("Created 18-second MP4, animated GIF, poster, and CLI provenance in", OUT)
