# Data

EndoSeg uses **public, anonymized** imaging data only. No patient/private data
is ever committed to this repo (see `.gitignore`).

## MMOTU (primary, MVP)

- **Dataset:** Multi-Modal Ovarian Tumor Ultrasound (MMOTU) — 2D ultrasound with
  pixel-level masks across 8 tumor classes, including endometrioma ("chocolate cyst").
- **Use here:** binary lesion-vs-background segmentation (MVP) + an endometrioma
  type flag. Full 8-class semantic segmentation is a stretch goal.
- **Download:** run `bash download_mmotu.sh` (see script header for required env vars).

### License — VERIFY BEFORE USE

> ⚠️ Confirm MMOTU's exact license terms before building on it. Verify it permits
> (a) research use and (b) redistribution of derived models. Record the confirmed
> license here once checked. Do not redistribute raw images if the license forbids it.

## GLENDA (stretch)

- **Dataset:** GLENDA — gynecologic laparoscopy endometriosis frames (2D, no slicing).
- **License note:** typically **CC BY-NC / research-only**. If used, cite it and
  keep usage non-commercial; display the research-only disclaimer in the app.

## Layout expected by the pipeline

```
data/
  raw/
    mmotu/
      images/   <id>.jpg
      masks/    <id>.png
  processed/    # produced by jobs/preprocess (resized, normalized, splits.json)
```

## Citation

Add full BibTeX entries for every dataset you use here and in the root `README.md`
Citation section before submission.
