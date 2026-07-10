# Deploying the EndoSeg browser demo

The demo is a static site under `browser/`. No build step is required beyond placing `model/unet.onnx` after export.

## Before you deploy

1. Export the browser ONNX model to `browser/model/unet.onnx`.
2. (Optional) Set the cloud-comparison proxy URL in `index.html`:

   ```html
   <meta name="proxy-url" content="https://your-proxy.example.com" />
   ```

   Without this, **Compare to cloud** stays disabled — local inference still works.

3. Regenerate sample thumbnails if needed:

   ```bash
   python browser/samples/generate_placeholders.py
   ```

---

## Option A — GitHub Pages

### `gh-pages` branch

```bash
# From repo root, copy browser/ contents to a deploy branch
git checkout --orphan gh-pages
git rm -rf .
cp -r browser/* .
git add .
git commit -m "Deploy EndoSeg browser demo"
git push origin gh-pages
```

In the repo **Settings → Pages**, set source to the `gh-pages` branch.

### `/docs` folder

1. Copy `browser/` to `docs/` at the repo root (or symlink in CI).
2. Enable **Pages → Deploy from branch → /docs**.

Site URL: `https://<user>.github.io/<repo>/`

---

## Option B — Netlify

1. Open [Netlify Drop](https://app.netlify.com/drop).
2. Drag the entire `browser/` folder onto the page.
3. Set **Publish directory** to `browser` if using Git-connected deploy.

Add environment-specific proxy URL by editing `index.html` before upload, or use a small build script to inject the meta tag.

---

## Option C — Local demo

```bash
npx serve browser
# open http://localhost:3000
```

Or with Python:

```bash
python -m http.server 8000 --directory browser
```

**Do not** open `index.html` via `file://` — Web Workers and ONNX fetch require HTTP.

---

## Checklist after deploy

- [ ] Model loads (or shows a clear error if `unet.onnx` is missing)
- [ ] Sample gallery images visible
- [ ] Privacy banner and footer disclaimer visible
- [ ] Mobile layout OK at 375px width
- [ ] Proxy meta tag set if cloud comparison is required
