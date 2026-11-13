# Demo media

- `demo.mp4`: 18 seconds, 1280 x 720, H.264, 20 fps, silent.
- `demo.gif`: 18-second looping README preview, 960 x 540.
- `demo-poster.png`: static alternative.
- `demo-provenance.json`: captured CLI commands, exit codes, and full reports.

The walkthrough is edited from actual CLI JSON results. It uses handwritten
fixtures from `examples/`, not real model captures, and its pacing does not
represent benchmark runtime. The commands shown across multiple visual lines
are one command, not PowerShell scripts to paste line by line.

Regenerate with `python scripts/render_demo.py` using an environment containing
Pillow, imageio, imageio-ffmpeg, and numpy. These are media-generation dependencies,
not runtime dependencies of llm-drift-bench. The script invokes the local source,
checks pass/fail exit codes, and saves the captured data before rendering.

The README embeds a GIF because it renders directly on GitHub. Its MP4 link
opens the repository's video file. For a native inline video player, the
maintainer can upload the MP4 through GitHub's Markdown editor and use the
resulting attachment URL. No attachment upload is performed by this script.
