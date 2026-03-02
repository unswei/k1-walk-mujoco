from pathlib import Path

import mujoco

from k1_walk_mujoco.assets.paths import K1_MJCF_PATH


def test_assets_present_and_loadable() -> None:
    assert K1_MJCF_PATH.exists(), (
        "Expected assets/booster/robots/K1/K1_22dof.xml to exist. "
        "Run python scripts/fetch_assets.py first."
    )
    model = mujoco.MjModel.from_xml_path(str(K1_MJCF_PATH))
    assert model.njnt > 0
