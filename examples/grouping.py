"""Minimal upper-level grouping loop; no desktop or recording dependency."""
from had_env import make_env
from had_env.grouping.policies import RulePolicy


def main():
    env = make_env(api="grouping", red=4, blue=4, targets=2,
                   task_mode="damage", max_steps=50)
    policy = RulePolicy("rule", seed=43)
    total_reward = 0.0
    try:
        state = env.reset(seed=42)
        while not env.done:
            state, reward, done, info = env.step(policy.act(state))
            total_reward += reward
        print(f"physical_steps={state.step}, defender_return={total_reward:.6f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
