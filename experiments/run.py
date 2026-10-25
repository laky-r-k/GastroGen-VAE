#!/usr/bin/env python3
"""
Experiment Runner CLI.
Unified entry point to run config-driven machine learning experiments.

Usage:
    python run.py --exp experiment1
    python run.py --exp experiment1 --config custom_config.yaml
    python run.py --list
"""

import os
import sys
import argparse
import importlib
import inspect
from typing import Dict, Type

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.base_experiment import BaseExperiment


def discover_experiments() -> Dict[str, Type[BaseExperiment]]:
    """
    Dynamically discovers all BaseExperiment subclasses in the experiments/ directory.
    """
    experiments_dir = os.path.join(PROJECT_ROOT, "experiments")
    registry: Dict[str, Type[BaseExperiment]] = {}

    for fname in os.listdir(experiments_dir):
        if fname.startswith("experiment") and fname.endswith(".py"):
            mod_name = f"experiments.{fname[:-3]}"
            try:
                mod = importlib.import_module(mod_name)
                for attr_name in dir(mod):
                    attr = getattr(mod, attr_name)
                    if (
                        inspect.isclass(attr)
                        and issubclass(attr, BaseExperiment)
                        and attr is not BaseExperiment
                    ):
                        key = fname[:-3]  # e.g., 'experiment1'
                        registry[key] = attr
            except Exception as e:
                print(f"⚠️ Warning: Could not inspect module {mod_name}: {e}")

    return registry


def main():
    parser = argparse.ArgumentParser(
        description="Unified Experiment Runner for Medical ML Benchmark"
    )
    parser.add_argument(
        "--exp",
        "-e",
        type=str,
        help="Name of the experiment to run (e.g. experiment1)",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Optional path to custom YAML config file (default: experiments/configs/<exp>.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Custom output directory to store results",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Disable timestamped subdirectories; write directly to results/<exp>",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all registered experiments and exit",
    )

    args = parser.parse_args()
    experiments_registry = discover_experiments()

    if args.list:
        print("\n📋 Available Experiments:")
        for exp_name, exp_cls in sorted(experiments_registry.items()):
            doc = (inspect.getdoc(exp_cls) or "No description").split("\n")[0]
            print(f"  • {exp_name:<16} - {doc}")
        print()
        sys.exit(0)

    if not args.exp:
        parser.print_help()
        sys.exit(1)

    exp_key = args.exp.lower().strip()
    if exp_key not in experiments_registry:
        print(f"❌ Error: Experiment '{args.exp}' not recognized.")
        print(f"   Available options: {list(experiments_registry.keys())}")
        sys.exit(1)

    exp_class = experiments_registry[exp_key]

    # Instantiate and run
    experiment_instance = exp_class(
        config_path=args.config,
        output_dir=args.output_dir,
        timestamped=not args.no_timestamp,
    )

    experiment_instance()


if __name__ == "__main__":
    main()
