# JoAPR: Cleaning the Lens of Prompt Learning for Vision-Language Models

This repository contains PyTorch implementation for CVPR2024 paper __JoAPR: Cleaning the Lens of Prompt Learning for Vision-Language Models__

[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Guo_JoAPR_Cleaning_the_Lens_of_Prompt_Learning_for_Vision-Language_Models_CVPR_2024_paper.pdf)

## How to Install

This code is built on top of the [CoOp](https://github.com/KaiyangZhou/CoOp). Please follow their steps to configure the runtime environment. Many thanks for their contributions!

Follow CoOp to install the datasets. Note that the [Food101N](https://www.kaggle.com/datasets/kuanghueilee/food-101n) dataset needs to be downloaded separately. [Food101N]([Food101N](https://www.kaggle.com/datasets/kuanghueilee/food-101n)) uses the same test set as Food101.

## How to Run

We provide the running scripts in `scripts/joapr`. You can adjust the hyperparameters in the config file in `configs/trainer/rn50.yaml`. Below we provide examples on how to run JoAPR on Caltech101.

**JoAPR(Caltech101, Symflip)**:

- 4 FP: `bash scripts/joapr/main.sh caltech101 4 symflip`
- 8 FP: `bash scripts/joapr/main.sh caltech101 8 symflip`
- 12 FP: `bash scripts/joapr/main.sh caltech101 12 symflip`

**JoAPR(Caltech101, Pairflip)**:

- 4 FP: `bash scripts/joapr/main.sh caltech101 4 pairflip`
- 8 FP: `bash scripts/joapr/main.sh caltech101 8 pairflip`
- 12 FP: `bash scripts/joapr/main.sh caltech101 12 pairflip`

To calculate the average results for the folder `rn50_16shots_4FP_symflip/nctx16_cscFalse_ctpend/`, you can run

```bash
python parse_test_res.py output/caltech101/rn50_16shots_4FP_symflip/nctx16_cscFalse_ctpend
```

## Citation

If you use this code in your research, please kindly cite the following paper.

```bash
@inproceedings{guo2024joapr,
  title={JoAPR: Cleaning the Lens of Prompt Learning for Vision-Language Models},
  author={Guo, Yuncheng and Gu, Xiaodong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={28695--28705},
  year={2024}
}
```

