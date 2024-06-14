# Classification Docker

### 依存環境

- Python 3.11.9
- CUDA 12.1.1
- torch==2.3.1
- torchvision==0.18.1
- pytorch-lightning==2.2.5
- torchmetrics==1.4.0.post0

### 環境構築(Docker)

## CLI 上で学習を行う場合

1\. コンテナの作成と実行

```
docker compose up -d
```

2\. コンテナのシェルを起動する

```
docker compose exec -it classification /bin/bash
```

3\. シェルを使って学習を実行する

例）

```
root@xxxxxxxxxx:/workspace# python src/train.py
```

4\. シェルから抜ける

```
exit
```

## Dev Containers 上で学習を行う場合

1\. コンテナの作成と実行

```
docker compose up
```

2\. リモートエクスプローラーの「開発コンテナー」を選択し、起動したコンテナにアタッチする

3\. VSCode 上でターミナルを表示し、学習を行う

### コンテナの停止

```
docker compose stop
```

再起動する際は以下のコマンドを実行する。

```
docker compose start
```

### コンテナの削除

```
docker compose down
```

## 学習

1\. データ格納先をバンドマウントするため[compose.yaml](./compose.yaml)の`volumes`を設定する

2\. 仮想環境を起動

```
docker compose up -d
```

3\. [config ファイル](./config/config.yaml)の パラメータを設定する

4\. 学習を実行する

```
python src/train.py
```

## 推論

1\. データ格納先をバンドマウントするため[compose.yaml](./compose.yaml)の`volumes`を設定する

2\. 仮想環境を起動

```
docker compose up -d
```

3\. 推論を実行する

```
python src/inference.py -i <path/to/image/root> -d <path/to/dataframe> -o <path/to/output/directory> -c <path/to/config/path> -m <path/to/model> -lmp <path/to/label map> -wrp <wandb run path> -wdp <wandb download root>
```

- 引数
  - `-i`, `--image_root`: 画像のルートパス
  - `-d`, `--df_path`: 推論用データフレームのパス
  - `-o`, `--output_dir`: 出力用ディレクトリのパス。デフォルトは`./outputs`
  - `-c`, `--config_path`: コンフィグファイルのパス。デフォルトは`config/config.yaml`
  - `-m`, `--model_path`: モデルファイルのパス
  - `-lmp`, `--label_map_path`: ラベルマップのパス
  - `-wrp`, `--wandb_run_path`: WandB 上のファイルを使う場合、WandB の run path を指定する
  - `-wdp`, `--wandb_download_root`: WandB 上のファイルを使う場合、ルートパスを指定する。デフォルトは`.`
  - `--tta`: Test Time Augmentation を行う
