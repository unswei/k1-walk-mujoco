import numpy as np

from k1_walk_mujoco.envs.k1_walk_env import K1WalkEnv


def test_env_step_no_nan_and_shape_consistent() -> None:
    env = K1WalkEnv()
    obs, _ = env.reset(seed=123)

    assert obs.shape == env.observation_space.shape

    for _ in range(5):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, rew, term, trunc, _ = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert np.isfinite(obs).all()
        assert np.isfinite(rew)
        if term or trunc:
            break

    env.close()


def test_env_deterministic_given_seed_and_actions() -> None:
    env1 = K1WalkEnv()
    env2 = K1WalkEnv()

    obs1, _ = env1.reset(seed=777)
    obs2, _ = env2.reset(seed=777)
    assert np.allclose(obs1, obs2)

    action = np.zeros(env1.action_space.shape, dtype=np.float32)
    for _ in range(5):
        obs1, rew1, term1, trunc1, _ = env1.step(action)
        obs2, rew2, term2, trunc2, _ = env2.step(action)
        assert np.allclose(obs1, obs2)
        assert np.isclose(rew1, rew2)
        assert term1 == term2
        assert trunc1 == trunc2

    env1.close()
    env2.close()
