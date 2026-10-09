"""Keep grouping and native flight results under separate protocols."""
from skyhad_workbench.protocols import ScenarioSpec
from skyhad_workbench.session import SimulationSession


def main():
    with SimulationSession(ScenarioSpec(red_count=8, blue_count=8), "rule") as session:
        episode = session.run()
        path = session.save("outputs/grouping-example.json.gz")
        print(f"frames={len(episode.frames)}, saved={path}")


if __name__ == "__main__":
    main()
