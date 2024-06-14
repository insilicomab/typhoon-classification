import json
from argparse import ArgumentParser
from pathlib import Path

from omegaconf import OmegaConf

import wandb
from dataset.dataset import get_inference_dataloader
from prediction.predictor import inference, load_model_weights, tta_inference


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-i", "--image_root", type=str, required=True)
    parser.add_argument("-d", "--df_path", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, default="./outputs")
    parser.add_argument("-c", "--config_path", type=str, default="config/config.yaml")
    parser.add_argument("-m", "--model_path", type=str, required=True)
    parser.add_argument(
        "-lmp", "--label_map_path", type=str, default="outputs/label_map.json"
    )
    parser.add_argument("-wrp", "--wandb_run_path", type=str, default=None)
    parser.add_argument("-wdp", "--wandb_download_root", type=str, default=".")
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()
    return args


def main(args):
    if args.wandb_run_path:
        print(f"Loading files from wandb run: {args.wandb_run_path}")
        api = wandb.Api()
        run = api.run(args.wandb_run_path)
        run.file(args.config_path).download(replace=True, root=args.wandb_download_root)
        run.file(args.model_path).download(replace=True, root=args.wandb_download_root)
        run.file(args.label_map_path).download(
            replace=True, root=args.wandb_download_root
        )

    # read config file
    config = OmegaConf.load(Path(args.wandb_download_root) / Path(args.config_path))

    # read label_map and generate int_to_label
    with open(args.label_map_path, "r") as f:
        label_map = json.load(f)
    int_to_label = {v: k for k, v in label_map.items()}

    # model
    model = load_model_weights(config, args.wandb_download_root, args.model_path)

    # dataloader
    dataloader = get_inference_dataloader(
        root=args.image_root,
        df_file_path=args.df_path,
        image_size=config.image_size,
    )

    if args.tta:
        print("Test Time Augmentation is Running")
        # Test Time Augmentation
        prediction_df = tta_inference(dataloader, model, int_to_label)
        prediction_df.to_csv("outputs/tta_inference_results.csv", index=False)

    else:
        # inference
        prediction_df = inference(dataloader, model, int_to_label)
        prediction_df.to_csv("outputs/inference_results.csv", index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
