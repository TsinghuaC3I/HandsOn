import numpy as np

from handson.tasks.mappings import CONFIG_MAPPING


def zero_action(env):
    return np.zeros(env.action_space.shape, dtype=env.action_space.dtype)


def main():
    for env_name, config_cls in CONFIG_MAPPING.items():
        print(f"Testing environment: {env_name}")
        config = config_cls()
        env = config.get_environment()
        action = zero_action(env)

        try:
            env.reset()
            for _ in range(100):
                _, _, done, truncated, _ = env.step(action)
                if done or truncated:
                    env.reset()
            print(f"{env_name}: ok")
        finally:
            env.close()


if __name__ == "__main__":
    main()
