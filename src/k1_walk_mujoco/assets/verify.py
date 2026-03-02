from pathlib import Path

from k1_walk_mujoco.assets.paths import K1_MJCF_PATH


def ensure_k1_assets_present() -> Path:
    if not K1_MJCF_PATH.exists():
        raise FileNotFoundError(
            f"Missing asset file: {K1_MJCF_PATH}. Run: python scripts/fetch_assets.py"
        )
    return K1_MJCF_PATH
