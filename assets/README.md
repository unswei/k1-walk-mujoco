# Assets

This project uses robot assets from [Booster Robotics booster_assets](https://github.com/BoosterRobotics/booster_assets).

## Attribution

- Source: Booster Robotics
- Repository: `https://github.com/BoosterRobotics/booster_assets`

## What is downloaded

`assets/manifest.json` pins a repository revision and file list. For v0 we download:

- `robots/K1/K1_22dof.xml`
- XML-referenced dependencies (for example `robots/K1/meshes/*.STL`)

Files are copied to `assets/booster/`.

## Refreshing assets

1. Update `rev` in `assets/manifest.json`.
2. Re-run:

```bash
python scripts/fetch_assets.py
```

This rewrites `assets/booster/.source.json` with source provenance and fetch timestamp.
