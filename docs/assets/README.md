# Demo & Documentation Assets

Static previews, animated walkthroughs, and sample reports referenced by the repository [README](../../README.md).

## Files in this directory

| File | Description |
| :--- | :--- |
| `demo.mp4` | 18-second silent walkthrough (1280 × 720, H.264, 20 fps). |
| `demo.gif` | 18-second looping README preview (960 × 540) rendered directly on GitHub. |
| `demo-poster.png` | Static poster alternative for readers who prefer no animation. |
| `demo-provenance.json` | Captured CLI commands, exit codes, and full JSON reports used to render the demo. |
| `example-report.html` | Self-contained offline HTML report generated from the bundled example fixtures (no external assets or JS). |
| `github-issues.png` | Screenshot asset used in repository documentation. |

## Provenance & regeneration

The walkthrough is edited from actual CLI JSON results on the handwritten fixtures in `examples/` — not real model captures or runtime benchmarks. Multi-line commands shown in the animation represent single CLI invocations.

To regenerate the video and GIF locally (requires media-only dependencies `Pillow`, `imageio`, `imageio-ffmpeg`, and `numpy`, which are **not** runtime dependencies of `llm-drift-bench`):

```bash
python scripts/render_demo.py
```

The script invokes the local source package, verifies pass/fail exit codes, saves `demo-provenance.json`, and renders the media files without performing any external uploads.
