from pathlib import Path

import pandas as pd
import torch
import ttach as tta
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.transformation import get_tta_transforms
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
    dataloader: DataLoader, model: torch.nn.Module, is_tta: bool = False
) -> pd.DataFrame:
    if CUDA_IS_AVAILABLE:
        model.cuda()

    if is_tta:
        print("Test Time Augmentation is Running")
        tta_transforms = get_tta_transforms()
        model = tta.ClassificationTTAWrapper(model, tta_transforms)

    paths, preds = [], []
    with torch.no_grad():
        for image, path, _ in tqdm(dataloader):
            if CUDA_IS_AVAILABLE:
                image = image.cuda()
            logits = model(image)
            pred = logits.argmax(dim=1)
            pred = pred.cpu().detach().numpy()
            paths.extend(path)
            preds.extend(pred)
    df = pd.DataFrame(
        {
            "filename": paths,
            "label": preds,
        }
    )
    return df
