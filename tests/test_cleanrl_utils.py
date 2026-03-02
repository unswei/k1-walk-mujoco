from __future__ import annotations

import pytest

from k1_walk_mujoco.rl.cleanrl import utils


def test_select_device_auto_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utils, "_is_cuda_available", lambda: True)
    monkeypatch.setattr(utils, "_is_mps_available", lambda: True)
    assert utils.select_device("auto").type == "cuda"


def test_select_device_auto_falls_back_to_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utils, "_is_cuda_available", lambda: False)
    monkeypatch.setattr(utils, "_is_mps_available", lambda: True)
    assert utils.select_device("auto").type == "mps"


def test_select_device_auto_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utils, "_is_cuda_available", lambda: False)
    monkeypatch.setattr(utils, "_is_mps_available", lambda: False)
    assert utils.select_device("auto").type == "cpu"


def test_select_device_unavailable_cuda_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utils, "_is_cuda_available", lambda: False)
    with pytest.raises(RuntimeError):
        utils.select_device("cuda")


def test_default_num_envs_by_platform() -> None:
    assert utils.default_num_envs(device_type="cuda", system_name="Linux") == 16
    assert utils.default_num_envs(device_type="mps", system_name="Darwin") == 8
    assert utils.default_num_envs(device_type="cpu", system_name="Darwin") == 4
    assert utils.default_num_envs(device_type="cpu", system_name="Linux") == 8


def test_resolve_num_envs_auto_and_non_positive() -> None:
    expected_cpu_default = utils.default_num_envs(device_type="cpu")
    assert utils.resolve_num_envs("auto", device_type="cpu") == expected_cpu_default
    assert utils.resolve_num_envs(0, device_type="cpu") == expected_cpu_default
    assert utils.resolve_num_envs(12, device_type="cuda") == 12
