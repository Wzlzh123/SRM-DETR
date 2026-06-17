# SRM-DETR: A Super-Resolution-Guided RGB–IRMultimodal Fusion Detector with Dual-AxisDeformable Convolution

![Paper Overview](docs/overview.jpg)

## Requirements
We use PaddlePaddle2.5(Stable) with the CUDA11.7 Linux version and our python version is 3.8. Please refer to the official guide of [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/tree/develop) for installation guide.

## Data Preparation
We provide annotated JSON files for VEDAI dataset in the `dataset` folder. You need to put VEDAI images in the `dataset/VEDAI512` directory according to the `train.json` and `val.json`.

## Run Code
Train on **VEDAI SR**:

```bash
nohup python tools/train.py -c configs/damsdet/damsdet_r50vd_vedai_SR.yml -o pretrain_weights=coco_pretrain_weights.pdparams --eval > srmdetr_vedai_SR_Pd.txt 2>&1 &
```

Evaluate:

```bash
python tools/eval.py -c configs/damsdet/damsdet_r50vd_vedai_SR.yml --classwise -o weights=output/VEDAI512/damsdet_r50vd_vedai_SR/best_model
```

## Acknowledgement
For the implementation, we rely on [Paddle](https://github.com/PaddlePaddle/Paddle) and [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/tree/develop).
