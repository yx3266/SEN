## [Support or Refute: Analyzing the Stance of Evidence to Detect Out-of-Context Mis- and Disinformation](https://aclanthology.org/2023.emnlp-main.259.pdf)

This repository contains all the code and instructions for reproducing the results of the paper. It is built on the top of [CCN](https://github.com/S-Abdelnabi/OoC-multi-modal-fc?tab=readme-ov-file).

To detect out-of-context (OOC) mis-/disinformation, we propose a unified framework that aims to comprehensively incorporate the stances of multiple pieces of evidence towards claims. More specifically, for image claim and evidence, caption claim and evidence, we utilize different but independent stance extraction networks (SENs) with a similar structure, which allow for cluster-specific presentations of evidence semantics and can extract and fuse multiple stances. In the textual SEN, we further emphasize the stance relationship through a support-refutation score calculated based on the co-occurrence relationship of named entities. The architecture of our proposed method is as follows:

<p align="center">
<img src="architecture.svg">
</p>



## Requirements

* Python 3.7.15
* Pytorch 1.7.1
* transformers 4.24.0
* spacy 3.3.0
* scipy 1.7.3
* numpy 1.21.5



## Usage

### 1. Organize the code and data

This repo and code are built based on the top of [CCN](https://github.com/S-Abdelnabi/OoC-multi-modal-fc?tab=readme-ov-file). So to reproduce our results, first organize the code and all the datasets (VisualNews, NewsCLIPpings and the evidence) according to CCN's requirements, and extract all necessary features.

### 2. Generate necessary features

#### 2.1 Support-Refutation Score

To generate SRS for textual evidence, run the following command under `data_preprocessing\`. Need to calculate for all splits.

```
python precompute_entity.py --split test # perform named entity recognition
python compute_srscore.py --split test # compute support-refutation score
```

#### 2.2 The evidence clusters in Stance Extraction Network

We calculate each evidence cluster before training, rather than during the training process, to speed up the training process.

To generate SuC/ReC/CoC in SEN, run the following command under `data_preprocessing\`. Need to calculate for all splits.

```
python compute_cap_cluster.py --split test   # for textual evidence of sentence type.
python compute_ent_cluster.py --split test   # for textual evidence of entity type, not used in ours.
python compute_img_cluster.py --split test   # for visual evidence.
```

If any questions or want to skip this step, please contact yuanxin@sjtu.edu.cn to obtain the generated features.

### 3. Train and Evaluate

#### 3.1 Train

To reproduce the best results reported in the paper, run the following command under `training_and_evaluation\sent_emb\`.

```
python train.py --mode train --ner_ent srscore --ner_cap srscore --cap_cluster --img_cluster --epochs 60
```

Although SEN is not used for textual evidence of entity type in the paper, it can be achieved by adding `--ent_cluster` to the command. In our testing, this brings an extremely small performance gain.

The results in `Ablation Analysis` only require adjustments to the parameters of the command.

* w/o SRS. Remove `--ner_ent srscore --ner_cap srscore`
* binary NEI. Use `--ner_ent binary --ner_cap binary`
* w/o Vi-SEN. Remove `--img_cluster`
* w/o Te-SEN. Remove `--img_cluster`
* w/o SENs. Remove `--cap_cluster --img_cluster`

#### 3.2 Evaluate

To evaluate the trained model,  run the following command under `training_and_evaluation\sent_emb\`.

```
python train.py --mode evaluate --ner_ent srscore --ner_cap srscore --cap_cluster --img_cluster --epochs 60
```

We have shared the trained models training logs, and detailed test results on the test set, which can be found in this [Google Drive link](https://drive.google.com/drive/folders/1-kmF4mm48Lpmsb8foq309I6aM_JHma_C?usp=drive_link).



## Citation

If you find our paper useful for your research, please include the following citation.

```
@inproceedings{yuan2023support,
  title={Support or Refute: Analyzing the Stance of Evidence to Detect Out-of-Context Mis- and Disinformation},
  author={Yuan, Xin and Guo, Jie and Qiu, Weidong and Huang, Zheng and Li, Shujun},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages={4268--4280},
  year={2023}
}
```

