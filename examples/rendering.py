"""User-owned action loop with native rendering or streaming video.

Install ``.[render]`` for projections, ``.[render,viewer3d]`` for 3D, and
``.[render,record]`` for MP4 (combine all three extras for a 3D video).

    python examples/rendering.py --model UAV_fixedwing --view 3d
    python examples/rendering.py --view projections --video episode.mp4

Rendering and capture read the current state. Only the explicit ``env.step``
below advances the environment. The caller owns the loop and env lifetime.
"""
import argparse
from contextlib import nullcontext
from pathlib import Path

from had_env import make_env
from had_env.render.video import VideoRecorder


def _positive_steps(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("max-steps must be positive")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", choices=("projections", "3d"), default="projections")
    parser.add_argument("--video", type=Path, help="Stream rendered RGB frames to this video")
    parser.add_argument("--model", choices=("particle", "UAV_fixedwing", "UAV_quadrotor"), default="UAV_fixedwing")
    parser.add_argument("--max-steps", type=_positive_steps, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    env = make_env("defense", red_count=4, blue_count=4, max_cycles=args.max_steps,
                   env_agent_type=args.model, spatial_dim=3,
                   render_mode="rgb_array" if args.video else "human")
    try:
        env.set_view_mode(args.view)
        env.reset(seed=args.seed)
        context = VideoRecorder(env, args.video) if args.video else nullcontext()
        with context as video:
            if video is not None:
                video.capture()
            for _ in range(args.max_steps):
                if not env.agents:
                    break
                actions = {agent: env.action_space(agent).sample() for agent in env.agents}
                env.step(actions)
                if video is not None:
                    video.capture()
                else:
                    env.render()
        print(f"steps={env.num_cycles}, view={args.view}" + (f", video={args.video}" if args.video else ""))
    finally:
        env.close()


if __name__ == "__main__":
    main()
