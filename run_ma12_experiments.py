import argparse

from npe_convergence.scripts.run_experiment import run_experiment
from npe_convergence.scripts.run_mak import run_mak


def run_mak_experiments(args):
    seed = args.seed
    _ = run_experiment(run_mak, seed, ma_order=12)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run_mak_experiments.py",
        description="Run experiments for mak model.",
        epilog="Example usage: python run_mak_experiments.py"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_mak_experiments(args)
