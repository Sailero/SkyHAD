"""Minimal 2D environment rendering; install had-env[render] first."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['particle', 'UAV_fixedwing', 'UAV_quadrotor'], default='UAV_fixedwing')
    parser.add_argument('--task', choices=['survival', 'damage'], default='survival')
    parser.add_argument('--view', choices=['projections'], default='projections')
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    from had_env import make_env
    env = make_env(env_agent_type=args.model, task_mode=args.task, render_mode='human', max_cycles=args.max_steps)
    env.set_view_mode(args.view)
    try:
        env.reset(seed=args.seed)
        for index, agent in enumerate(env.possible_agents):
            env.action_space(agent).seed(args.seed + index)
        while env.agents:
            env.step({a: env.action_space(a).sample() for a in env.agents})
    finally:
        env.close()


if __name__ == '__main__':
    main()
