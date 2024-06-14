from pathlib import Path

import pandas as pd
import torch
import ttach as tta
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.transformation import get_tta_transforms
from model.evidences import calculate_uncertainty
from model.model import get_model

CUDA_IS_AVAILABLE = torch.cuda.is_available()


def load_model_weights(
    config: DictConfig, download_root: str, model_path: str
) -> torch.nn.Module:
    _model_state_dict = torch.load(Path(download_root) / Path(model_path))["state_dict"]
    model_state_dict = {
        k.replace("model.", ""): v for k, v in _model_state_dict.items()
    }
    model = get_model(config)
    model.load_state_dict(model_state_dict, strict=True)
    model.eval()
    return model


def inference(
    dataloader: DataLoader, model: torch.nn.Module, int_to_label: dict
) -> pd.DataFrame:
    if CUDA_IS_AVAILABLE:
        model.cuda()
    paths, preds, uncertainties = [], [], []
    with torch.no_grad():
        for image, path, _ in tqdm(dataloader):
            if CUDA_IS_AVAILABLE:
                image = image.cuda()
            logits = model(image)
            pred = logits.argmax(dim=1)
            pred = pred.cpu().detach().numpy()
            pred = [int_to_label[i] for i in pred]
            uncertainty = calculate_uncertainty(logits, num_classes=len(int_to_label))
            paths.extend(path)
            preds.extend(pred)
            uncertainties.extend(uncertainty)
    df = pd.DataFrame(
        {
            "image_id": paths,
            "label": preds,
            "uncertainty": uncertainties,
        }
    )
    return df


def tta_inference(
    dataloader: DataLoader, model: torch.nn.Module, int_to_label: dict
) -> pd.DataFrame:
    # transforms for tta
    tta_transforms = get_tta_transforms()

    if CUDA_IS_AVAILABLE:
        model.cuda()
    tta_model = tta.ClassificationTTAWrapper(model, tta_transforms)

    paths, preds, uncertainties = [], [], []
    with torch.no_grad():
        for image, path, _ in tqdm(dataloader):
            if CUDA_IS_AVAILABLE:
                image = image.cuda()
            logits = tta_model(image)
            pred = logits.argmax(dim=1)
            pred = pred.cpu().detach().numpy()
            pred = [int_to_label[i] for i in pred]
            uncertainty = calculate_uncertainty(logits, num_classes=len(int_to_label))
            paths.extend(path)
            preds.extend(pred)
            uncertainties.extend(uncertainty)
    df = pd.DataFrame(
        {
            "image_id": paths,
            "label": preds,
            "uncertainty": uncertainties,
        }
    )
    return df
