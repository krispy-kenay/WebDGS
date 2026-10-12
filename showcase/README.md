# WebDGS showcase bundle

Web-ready media and a machine-readable manifest describing WebDGS, so other sites
(e.g. the portfolio at yannickgachnang.com) can pull the project in without
hand-copying assets. Everything is static; paths in `showcase.json` are relative
to this folder.

## Layout

```
showcase/
├── showcase.json      manifest (schema below)
├── README.md          this file
├── video/             H.264 MP4 (+faststart) and VP9 WebM, muted, 980×544
├── poster/            JPG + WebP poster frame per video (last frame, 980×544)
├── gif/               small GIF fallbacks (also embedded in the root README)
└── image/             stills (JPG + WebP)
```

Media IDs: `locomotive-flythrough`, `training-side-view`, `training-front-view`,
`training-cab-closeup`, `photo-vs-webdgs-20k`. Each ID is the base file name in
every folder.

## Manifest schema (`schema_version: 1`)

Top level:

| Field | Type | Notes |
| --- | --- | --- |
| `schema_version` | int | Bumped on breaking changes. |
| `id` | string | Stable project slug (`webdgs`). |
| `title`, `subtitle`, `blurb` | string | Display text. |
| `demo_url` | URL | Live GitHub Pages build. |
| `repo_url` | URL | Source repository. |
| `tags` | string[] | Short labels. |
| `cover` | string | `id` of the media entry to use as the card cover. |
| `media` | object[] | Ordered: render in array order. |

Media entry:

| Field | Type | Notes |
| --- | --- | --- |
| `id` | string | Unique within the manifest. |
| `type` | `"video"` \| `"image"` | |
| `title`, `caption`, `alt` | string | `alt` is screen-reader text; `caption` is visible text. |
| `files` | object | Format → relative path. Videos: `mp4`, `webm`, `gif`. Images: `jpg`, `webp`. |
| `poster` | object | Videos only. `jpg`, `webp`. |
| `width`, `height` | int | Pixel size of `files` (and poster). GIFs are smaller: see `gif_width`, `gif_height`. |
| `duration_s` | number | Videos only. |
| `autoplay`, `loop`, `muted` | bool | Videos only; the files carry no audio track. |
| `bytes`, `poster_bytes` | object | File sizes per format, for budgeting. |

Example entry:

```json
{
  "id": "training-side-view",
  "type": "video",
  "title": "Training progress: side view",
  "files": {
    "mp4": "video/training-side-view.mp4",
    "webm": "video/training-side-view.webm",
    "gif": "gif/training-side-view.gif"
  },
  "poster": { "jpg": "poster/training-side-view.jpg", "webp": "poster/training-side-view.webp" },
  "caption": "Side view of the locomotive as training runs from iteration 0 (sparse COLMAP points) to a dense splat scene.",
  "alt": "Side view of a green locomotive turning from scattered noisy points into a sharp image as the iteration and Gaussian counters climb.",
  "width": 980, "height": 544, "gif_width": 360, "gif_height": 200,
  "duration_s": 11.7, "autoplay": true, "loop": true, "muted": true,
  "bytes": { "mp4": 1692112, "webm": 1590969, "gif": 2757890 },
  "poster_bytes": { "jpg": 78240, "webp": 53374 }
}
```

Suggested markup for a video entry:

```html
<video autoplay loop muted playsinline preload="none"
       width="980" height="544" poster="poster/training-side-view.webp"
       aria-label="(alt text)">
  <source src="video/training-side-view.webm" type="video/webm">
  <source src="video/training-side-view.mp4" type="video/mp4">
</video>
```

## Pulling it into another repo with GitHub Actions

The folder is self-contained, so a consumer can check out just `showcase/`
with a sparse checkout and copy it into its own tree. Example workflow for the
portfolio repo:

```yaml
name: Sync WebDGS showcase
on:
  schedule: [{ cron: '0 4 * * 1' }]
  workflow_dispatch:
permissions:
  contents: write
jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/checkout@v4
        with:
          repository: krispy-kenay/WebDGS
          ref: main
          path: _webdgs
          sparse-checkout: showcase
      - name: Copy bundle
        run: |
          rm -rf projects/webdgs
          mkdir -p projects/webdgs
          cp -r _webdgs/showcase/. projects/webdgs/
          rm -rf _webdgs
          jq -e '.schema_version == 1' projects/webdgs/showcase.json
      - name: Commit if changed
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "41898283+github-actions[bot]@users.noreply.github.com"
          git add projects/webdgs
          git diff --cached --quiet || git commit -m "chore: sync WebDGS showcase" && git push
```

Pin `ref` to a tag or commit SHA if you want the portfolio to update only on
purpose. Check `schema_version` before rendering.

## Regenerating the media

Sources are the 980×544 screen captures. The encodes were made with:

```bash
# MP4 (H.264, faststart, no audio)
ffmpeg -i in.mp4 -an -c:v libx264 -preset slow -crf 26 -pix_fmt yuv420p -profile:v high -movflags +faststart out.mp4
# WebM (VP9, no audio)
ffmpeg -i in.mp4 -an -c:v libvpx-vp9 -crf 38 -b:v 0 -row-mt 1 -cpu-used 2 out.webm   # flythrough: -crf 42
# Poster (last frame)
ffmpeg -ss <duration-0.5> -i in.mp4 -frames:v 1 -q:v 3 poster.jpg
ffmpeg -ss <duration-0.5> -i in.mp4 -frames:v 1 -c:v libwebp -quality 80 poster.webp
# GIF fallback
ffmpeg -i in.gif -vf "fps=8,scale=360:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" out.gif
```

After changing a file, update `bytes` in `showcase.json`.
