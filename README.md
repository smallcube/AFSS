# Adaptive Feature Space Shrinking for Deep Neural Networks
Zhi Chen, Jiang Duan*, Uwe Aickelin, Ling Zhang, Xiyang Chen and Guoping Qiu

This repository is the official PyTorch implementation of the paper [AFSS](https://arxiv.org/abs/).

## Environments

```shell
pytorch >= 2.1.0
timm == 0.3.2
```

1. If your PyTorch is 1.8.0+, a [fix](https://github.com/huggingface/pytorch-image-models/issues/420) is needed to work with timm.


## Usage

1. You can see all our settings in ./config/

2. Typically, 2 GPUs and >=24 GB per GPU Memory are required to train the ResNet50. But when training ViT-B-16 with a training resolution of 384, bigger GPU Memory is required.

```python
python main_cnn_ensemble_ddp.py (or python main_vit_ensemble.py if you want to train ViT)

or

torchrun --nproc_per_node=n main_cnn_ensemble_ddp.py

where n is the number of gpus in your server. And you should divide the defaulting batch_size in our configs with n.
```



## Citation

If you find our idea or code inspiring, please cite our paper:

```bibtex
@article{CADEL,
  title={Adaptive Feature Space Shrinking for Deep Neural Networks},
  author={Zhi Chen, Jiang Duan, Uwe Aickelin, Ling Zhang, Xiyang Chen and Guoping Qiu},
  year={2024},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```
