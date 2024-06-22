"""
This code contains an Experiment Tracker class.
Specifically, it is a wrapper for wandb or mlflow.
"""

import os
from enum import Enum

import mlflow
import wandb
from pytorch_lightning.loggers import MLFlowLogger, WandbLogger


class Logger(Enum):
    WANDB = "wandb"
    MLFLOW = "mlflow"


class ExperimentTracker:
    """
    Experiment Tracker class
    """

    def __init__(self, config):
        self.config = config
        self.tracker = self.config.tracker
        self.tracker_name = self.tracker.name

        if self.tracker_name == Logger.WANDB.value:
            self.tracker_config = self.tracker.wandb
            self._init_wandb()
        elif self.tracker_name == Logger.MLFLOW.value:
            self.tracker_config = self.tracker.mlflow
            self._init_mlflow()
        else:
            raise ValueError(f"Invalid tracker type: {self.tracker_name}")

    def _init_wandb(self):
        wandb.init(
            project=self.tracker_config.project,
            name=self.tracker_config.run_name,
            tags=self.tracker_config.tags,
            notes=self.tracker_config.notes,
            config={
                "data": os.path.basename(self.tracker.data_dir),
                "model": self.tracker.model_name,
            },
        )
        self.wandb_logger = WandbLogger(project=self.tracker_config.project)

    def _init_mlflow(self):
        mlflow.set_tracking_uri(self.tracker_config.uri)
        mlflow.set_experiment(self.tracker_config.experiment)
        run = mlflow.start_run(run_name=self.tracker_config.run_name)
        mlflow.log_params(
            {
                "data": os.path.basename(self.tracker.data_dir),
                "model": self.tracker.model_name,
            }
        )
        self.mlflow_logger = MLFlowLogger(
            experiment_name=self.tracker_config.experiment,
            run_name=self.tracker_config.run_name,
            run_id=run.info.run_id,
        )

    @property
    def logger(self):
        if self.tracker_name == Logger.WANDB.value:
            return self.wandb_logger
        elif self.tracker_name == Logger.MLFLOW.value:
            return self.mlflow_logger

    def log_metrics(self, metrics):
        if self.tracker_name == Logger.WANDB.value:
            self.wandb_logger.log(metrics)
        elif self.tracker_name == Logger.MLFLOW.value:
            mlflow.log_metrics(metrics)

    def log_artifacts(self, artifact_path, base_path=None):
        if self.tracker_name == Logger.WANDB.value:
            wandb.save(artifact_path, base_path=base_path, policy="now")
        elif self.tracker_name == Logger.MLFLOW.value:
            mlflow.log_artifact(artifact_path)


class DummyExperimentTracker:
    """
    Dummy Experiment Tracker class
    """

    def __init__(self):
        pass

    def log_metrics(self, metrics):
        pass

    def log_artifacts(self, artifact_path, base_path=None):
        pass

    @property
    def logger(self):
        return None
