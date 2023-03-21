# Introduction

This is the source code of our TOMM 2023 paper "LFR-GAN: Local Feature Refinement based Generative Adversarial Network for Text-to-Image Generation". Please cite the following paper if you use our code.

Zijun Deng, Xiangteng He and Yuxin Peng*, Zijun Deng, Xiangteng He and Yuxin Peng*, "LFR-GAN: Local Feature Refinement based Generative Adversarial Network for Text-to-Image Generation", ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM), 2023.


# Dependencies

- Python 3.7

- CUDA 1.11.0

- PyTorch 1.7.1

- gcc 7.5.0

Run the following commands to install the same dependencies as our experiments.

```bash
conda install -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt
```


# Data Preparation

Download the image data, pretrained models, and parser tool that we used from the [link](https://pan.baidu.com/s/1Q9Vh2JTOTHnsjmKlyqum2g) (password: 2fsx) and unzip them to corresponding folders.


# Generate Image

1. For bird images: `python code/main.py --cfg=code/cfg/eval_bird.yml --lafite=pretrained_models/birds.pkl`

2. For flower images: `python code/main.py --cfg=code/cfg/eval_flower.yml  --lafite=pretrained_models/flower.pkl`


# Train Model

You can also train the models by yourself.
```bash
# pretrained_models/bird_netG_epoch_700.pth
python code/main_DMGAN.py --cfg code/cfg/bird_DMGAN.yml --gpu 0
# pretrained_models/flower_netG_epoch_325.pth
python code/main_DMGAN.py --cfg code/cfg/flower_DMGAN.yml --gpu 0
```

For the training of Lafite models, please refer to [https://github.com/drboog/Lafite](https://github.com/drboog/Lafite).

# Evaluate

For the evaluation, please refer to [Lafite](https://github.com/drboog/Lafite).


For any questions, feel free to contact us (dengzijun57@gmail.com).

Welcome to our [Laboratory Homepage](http://www.icst.pku.edu.cn/mipl/home/) for more information about our papers, source codes, and datasets.
