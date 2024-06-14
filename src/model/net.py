from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from dataset.transformation import get_cutmix_and_mixup
from model.loss import get_loss_fn
from model.metrics import (
    get_classification_metrics,
    get_classification_report,
    get_confusion_matrix,
    get_metrics,
    get_roc_curve,
)
from model.model import get_model
from model.optimizer import get_optimizer
from model.scheduler import get_scheduler


class Net(pl.LightningModule):
    def __init__(self, label_map: dict, config: DictConfig):
        super().__init__()

        self.config = config

        # label
        self.int_to_label = {v: k for k, v in label_map.items()}

        # model
        self.model = get_model(config=self.config)

        # loss function
        self.loss_fn = get_loss_fn(config=self.config)

        # metrics
        metrics = get_metrics(config=self.config)
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

        # cutmix / mixup
        self.cutmix_or_mixup = get_cutmix_and_mixup(config=self.config)

        # test outputs for evaluation
        self.test_outputs = {
            "true": [],
            "pred_proba": [],
            "pred": [],
            "true_onehot": [],
        }

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = get_optimizer(config=self.config, net=self.model)
        scheduler = get_scheduler(config=self.config, optimizer=optimizer)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, t = batch

        # CutMix / MixUp
        if (
            self.config.train_transform.cutmix_mixup.enable
            and self.current_epoch < self.config.train_transform.cutmix_mixup.max_epochs
        ):
            x, t = self.cutmix_or_mixup(x, t)

        y = self(x)

        loss = self.loss_fn(y, t)

        preds = F.softmax(y, dim=1)

        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        self.train_metrics(preds, t)
        self.log_dict(
            self.train_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        preds = F.softmax(y, dim=1)

        loss = self.loss_fn(y, t)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        self.val_metrics(preds, t)
        self.log_dict(
            self.val_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        y_pred_proba = F.softmax(y, dim=1)
        y_pred = torch.argmax(y, dim=1)
        t_onehot = torch.eye(self.config.num_classes)[t.to("cpu")]

        self.test_outputs["true"].append(t.to("cpu").squeeze())
        self.test_outputs["pred_proba"].append(y_pred_proba.to("cpu"))
        self.test_outputs["pred"].append(y_pred.to("cpu").squeeze())
        self.test_outputs["true_onehot"].append(t_onehot.to("cpu"))

    def on_test_epoch_end(self):
        true = np.array(self.test_outputs["true"])
        y_pred_proba = torch.cat(self.test_outputs["pred_proba"], dim=0)
        y_pred = np.array(self.test_outputs["pred"])
        t_onehot = torch.cat(self.test_outputs["true_onehot"], dim=0)

        self._evaluate((true, y_pred_proba, y_pred, t_onehot))

    def _evaluate(self, ys):
        """
        Evaluation pipeline.
        """
        true, y_proba, y_pred, t_onehot = ys
        self._log_metrics(true, y_proba, y_pred, t_onehot)
        self._save_confusion_matrix(true, y_pred)
        self._save_classification_report(true, y_pred)
        self._save_roc_curve(t_onehot, y_proba)

    def _log_metrics(self, true, y_proba, y_pred, t_onehot):
        results = get_classification_metrics(
            true, y_proba, y_pred, t_onehot, config=self.config
        )
        accuracy, precision, recall, f1, specificity, kappa, auc = results
        self.log("test_accuracy", accuracy)
        self.log("test_precision", precision)
        self.log("test_recall", recall)
        self.log("test_fbeta", f1)
        self.log("test_specifity", specificity)
        self.log("test_kappa", kappa)
        self.log("test_auc", auc)

    def _save_classification_report(self, true, y_pred):
        """
        Save classification report to txt.
        """
        cls_report_str = get_classification_report(true, y_pred, self.int_to_label)
        with open("outputs/classification_report.txt", "w") as f:
            f.write(cls_report_str)

    def _save_confusion_matrix(self, true, y_pred):
        """
        Save confusion matrix.
        """
        cm = get_confusion_matrix(
            true, y_pred, labels=np.arange(len(self.int_to_label))
        )
        df_cm = pd.DataFrame(
            cm, index=self.int_to_label.values(), columns=self.int_to_label.values()
        )
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, cmap="Blues", fmt="d")
        plt.xlabel("Prediction label")
        plt.ylabel("True label")
        plt.savefig("outputs/confusion_matrix.png")

    def _save_roc_curve(self, t_onehot, y_proba):
        """
        Save ROC curve
        """
        fpr, tpr, roc_auc = get_roc_curve(t_onehot, y_proba, self.config)

        plt.figure()
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})',
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )

        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label=f'macro-average ROC curve (area = {roc_auc["macro"]:0.2f})',
            color="navy",
            linestyle=":",
            linewidth=4,
        )

        colors = cycle(["aqua", "darkorange", "cornflowerblue", "seagreen", "tomato"])
        for i, color in zip(range(self.config.num_classes), colors):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=2,
                label=f"ROC curve of class {self.int_to_label[i]} (area = {roc_auc[i]:0.2f})",
            )

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Some extension of Receiver operating characteristic to multiclass")
        plt.legend(loc="lower right")
        plt.savefig("outputs/roc_curve.png")
