"""
Base Experiment Module.
Provides the foundational class for config-driven, reproducible, versioned ML experiments.
"""

import os
import sys
import yaml
import shutil
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List


class BaseExperiment(ABC):
    """
    Abstract Base Class for all machine learning experiments.

    Handles:
      - Config loading (from YAML or dict)
      - Timestamped, versioned result directories (results/{exp_name}/run_YYYY-MM-DD_HH-MM-SS/)
      - Config snapshotting for 100% reproducibility
      - Standard dataset loading and train/test feature extraction
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        timestamped: bool = True,
    ):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.timestamped = timestamped

        # 1. Resolve configuration
        if config is not None:
            self.config = config
            self.config_path = config_path
        elif config_path is not None:
            self.config_path = self._resolve_path(config_path)
            self.config = self._load_yaml(self.config_path)
        else:
            default_config_path = self._get_default_config_path()
            if os.path.exists(default_config_path):
                self.config_path = default_config_path
                self.config = self._load_yaml(self.config_path)
            else:
                raise ValueError(
                    f"No config provided and default config not found at: {default_config_path}"
                )

        self.experiment_name = self.config.get("experiment", self.__class__.__name__.lower())

        # 2. Setup output directories
        if output_dir is not None:
            self.run_dir = self._resolve_path(output_dir)
        else:
            now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if self.timestamped:
                self.run_dir = os.path.join(
                    self.project_root, "results", self.experiment_name, f"run_{now_str}"
                )
            else:
                self.run_dir = os.path.join(
                    self.project_root, "results", self.experiment_name
                )

        os.makedirs(self.run_dir, exist_ok=True)

        # 3. Save copy of config for reproducibility
        self._save_config_snapshot()

    def _resolve_path(self, path: str) -> str:
        """Resolves relative paths against the project root."""
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self.project_root, path))

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """Loads and parses a YAML configuration file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}

    def _get_default_config_path(self) -> str:
        """Determines default config path based on class name or experiment name."""
        name = getattr(self, "experiment_name", None)
        if not name:
            import re
            name = re.sub(r'(?<!^)(?=[A-Z])', '_', self.__class__.__name__).lower()
        
        # Check direct name or with underscores
        candidates = [
            os.path.join(self.project_root, "experiments", "configs", f"{name}.yaml"),
            os.path.join(self.project_root, "experiments", "configs", f"{self.__class__.__name__.lower()}.yaml"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return candidates[0]

    def _save_config_snapshot(self) -> None:
        """Saves a verbatim copy of the configuration into the run output directory."""
        dest_config_path = os.path.join(self.run_dir, "config.yaml")
        with open(dest_config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

    def get_results_subdir(self, subdir_name: str) -> str:
        """Creates and returns a sub-directory inside the current run folder."""
        sub_path = os.path.join(self.run_dir, subdir_name)
        os.makedirs(sub_path, exist_ok=True)
        return sub_path

    def load_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], pd.Series, pd.Series]:
        """
        Loads training and test datasets specified in self.config['dataset'].

        Returns:
            train_df, test_df, feature_cols, y_train, y_test
        """
        dataset_cfg = self.config.get("dataset", {})
        train_path = self._resolve_path(dataset_cfg["train_path"])
        test_path = self._resolve_path(dataset_cfg["test_path"])
        label_col = dataset_cfg.get("label_column", "label")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        feature_cols = [c for c in train_df.columns if c != label_col]
        y_train = train_df[label_col]
        y_test = test_df[label_col]

        return train_df, test_df, feature_cols, y_train, y_test

    @abstractmethod
    def run(self) -> Any:
        """
        Executes the experiment pipeline. Must be implemented by subclasses.
        """
        pass

    def __call__(self) -> Any:
        print("\n" + "=" * 70)
        print(f"🚀 Launching Experiment: {self.experiment_name}")
        print(f"📁 Run Directory:        {self.run_dir}")
        if hasattr(self, "config_path") and self.config_path:
            print(f"⚙️  Config File:          {self.config_path}")
        print("=" * 70 + "\n")

        start_time = datetime.now()
        results = self.run()
        elapsed = datetime.now() - start_time

        print("\n" + "=" * 70)
        print(f"✅ Finished: {self.experiment_name} in {elapsed}")
        print(f"📦 Results preserved in: {self.run_dir}")
        print("=" * 70 + "\n")
        return results
