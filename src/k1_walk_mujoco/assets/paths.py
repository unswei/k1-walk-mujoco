from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ASSETS_DIR = REPO_ROOT / "assets"
BOOSTER_DIR = ASSETS_DIR / "booster"
K1_MJCF_PATH = BOOSTER_DIR / "robots" / "K1" / "K1_22dof.xml"
MANIFEST_PATH = ASSETS_DIR / "manifest.json"
SOURCE_STAMP_PATH = BOOSTER_DIR / ".source.json"
