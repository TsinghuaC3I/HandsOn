from handson.sim.isaaclab.abc_collection_env import (
    supported_abc_collection_tasks,
    supported_native_collection_tasks,
)
from handson.tasks.mappings import CONFIG_MAPPING


NEW_C_TOOL_TASKS = (
    "t02_connect_charger_and_cable",
    "t11_line_up_blocks",
)


def test_new_c_tool_tasks_are_registered_for_isaac_collection() -> None:
    abc_tasks = supported_abc_collection_tasks()
    native_tasks = supported_native_collection_tasks()
    for task_name in NEW_C_TOOL_TASKS:
        assert task_name in CONFIG_MAPPING
        assert CONFIG_MAPPING[task_name].task_name == task_name
        assert task_name in native_tasks


def test_new_c_tool_task_configs_are_isaac_only() -> None:
    for task_name in NEW_C_TOOL_TASKS:
        config = CONFIG_MAPPING[task_name]()
        try:
            config.get_environment(sim_backend="mujoco", launch_isaac=False, image_obs=False)
        except ValueError as exc:
            assert "isaacsim" in str(exc)
        else:
            raise AssertionError(f"{task_name} accepted a non-Isaac backend")


def test_new_c_tool_task_configs_build_policy_env_without_isaac_launch() -> None:
    for task_name in NEW_C_TOOL_TASKS:
        config = CONFIG_MAPPING[task_name]()
        env = config.get_environment(
            policy_mode=True,
            sim_backend="isaacsim",
            launch_isaac=False,
            image_obs=False,
            render_mode="rgb_array",
        )
        try:
            assert env.observation_space["state"].shape[0] > 0
        finally:
            env.close()
