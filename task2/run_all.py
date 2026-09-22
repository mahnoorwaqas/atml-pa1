from tqdm import tqdm
import subprocess
import sys


def run(command, description):
    print(f"\n{'=' * 70}")
    print(description)
    print(f"{'=' * 70}")

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"\nFAILED: {description}")
        sys.exit(result.returncode)


def main():

    steps = [
        (
            "Training Source Only",
            [
                sys.executable,
                "task2/train.py",
                "--config",
                "task2/configs/source_only.yaml",
            ],
        ),
        (
            "Training DAN",
            [
                sys.executable,
                "task2/train.py",
                "--config",
                "task2/configs/dan.yaml",
            ],
        ),
        (
            "Training DANN",
            [
                sys.executable,
                "task2/train.py",
                "--config",
                "task2/configs/dann.yaml",
            ],
        ),
        (
            "Training CDAN",
            [
                sys.executable,
                "task2/train.py",
                "--config",
                "task2/configs/cdan.yaml",
            ],
        ),
        (
            "Final Evaluation",
            [
                sys.executable,
                "task2/evaluate_final.py",
                "--config",
                "task2/configs/base.yaml",
            ],
        ),
        (
            "Controlled Study (DAN)",
            [
                sys.executable,
                "task2/run_controlled_study.py",
                "--config",
                "task2/configs/dan.yaml",
            ],
        ),
    ]

    for description, command in tqdm(
        steps,
        desc="Task 2 pipeline",
        unit="stage"
    ):
        run(command, description)

    print("\nTASK 2 COMPLETE")


if __name__ == "__main__":
    main()