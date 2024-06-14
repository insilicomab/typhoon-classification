import torch
import ttach as tta
from omegaconf import DictConfig
from PIL import Image
from torchvision.transforms import v2


def get_cutmix_and_mixup(config: DictConfig):
    cutmix = v2.CutMix(
        num_classes=config.num_classes,
        alpha=config.train_transform.cutmix_mixup.mixup.alpha,
    )
    mixup = v2.MixUp(
        num_classes=config.num_classes,
        alpha=config.train_transform.cutmix_mixup.cutmix.alpha,
    )
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    return cutmix_or_mixup


def get_tta_transforms():
    return tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Scale(scales=[1, 2, 4]),
        ]
    )


class Nonetransform:
    def __call__(self, image):
        return image


class Transforms:
    def __init__(self, config: DictConfig) -> None:
        self.data_transform = {
            "train": v2.Compose(
                [
                    v2.ToImage(),
                    (
                        v2.RandomCrop(
                            (
                                config.train_transform.random_crop.image_size,
                                config.train_transform.random_crop.image_size,
                            )
                        )
                        if config.train_transform.random_crop.enable
                        else Nonetransform()
                    ),
                    (
                        v2.RandAugment(
                            config.train_transform.randaugment.num_ops,
                            config.train_transform.randaugment.magnitude,
                        )
                        if config.train_transform.randaugment.enable
                        else Nonetransform()
                    ),
                    (
                        v2.TrivialAugmentWide()
                        if config.train_transform.trivial_augment_wide.enable
                        else Nonetransform()
                    ),
                    (
                        v2.AugMix(
                            severity=config.train_transform.augmix.severity,
                            mixture_width=config.train_transform.augmix.mixture_width,
                            chain_depth=config.train_transform.augmix.chain_depth,
                            alpha=config.train_transform.augmix.alpha,
                            all_ops=config.train_transform.augmix.all_ops,
                        )
                        if config.train_transform.augmix.enable
                        else Nonetransform()
                    ),
                    v2.ToDtype(torch.float32, scale=True),
                    (
                        v2.Normalize(
                            config.train_transform.normalize.mean,
                            config.train_transform.normalize.std,
                        )
                        if config.train_transform.normalize.enable
                        else Nonetransform()
                    ),
                ]
            ),
            "val": v2.Compose(
                [
                    v2.ToImage(),
                    (
                        v2.CenterCrop(
                            (
                                config.test_transform.center_crop.image_size,
                                config.test_transform.center_crop.image_size,
                            )
                        )
                        if config.test_transform.center_crop.enable
                        else Nonetransform()
                    ),
                    v2.ToDtype(torch.float32, scale=True),
                    (
                        v2.Normalize(
                            config.test_transform.normalize.mean,
                            config.test_transform.normalize.std,
                        )
                        if config.test_transform.normalize.enable
                        else Nonetransform()
                    ),
                ]
            ),
            "test": v2.Compose(
                [
                    v2.ToImage(),
                    (
                        v2.CenterCrop(
                            (
                                config.test_transform.center_crop.image_size,
                                config.test_transform.center_crop.image_size,
                            )
                        )
                        if config.test_transform.center_crop.enable
                        else Nonetransform()
                    ),
                    v2.ToDtype(torch.float32, scale=True),
                    (
                        v2.Normalize(
                            config.test_transform.normalize.mean,
                            config.test_transform.normalize.std,
                        )
                        if config.test_transform.normalize.enable
                        else Nonetransform()
                    ),
                ]
            ),
        }

    def __call__(self, phase: str, img: Image) -> torch.Tensor:
        return self.data_transform[phase](img)


class TestTransforms:
    def __init__(self, image_size: int):
        self.data_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.CenterCrop((image_size, image_size)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, img: Image) -> torch.Tensor:
        return self.data_transform(img)


__all__ = ["Transforms", "TestTransforms", "get_cutmix_and_mixup"]
