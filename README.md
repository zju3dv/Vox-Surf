# Vox-Surf
Code for "Vox-Surf: Voxel-based Implicit Surface Representation", TVCG 2022


## Installation

It is recommended to install [Pytorch](https://pytorch.org/get-started/locally/) (>=1.10) manually for your hardware platform first. You can then install all dependancies using `pip` or `conda`:

```
pip install -r requirements.txt
```

After you have installed all third party libraries, build extra Pytorch module as follows.

```bash
python setup.py install
```

## Run

Please follow [NeuS](https://github.com/Totoro97/NeuS) for preparing training data and follow `run.sh` for training and evaluation. 


## Citation

```
@article{li2022vox,
  title={Vox-Surf: Voxel-based implicit surface representation},
  author={Li, Hai and Yang, Xingrui and Zhai, Hongjia and Liu, Yuqian and Bao, Hujun and Zhang, Guofeng},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2022}
}
```

## Acknowledgement

Some code snippets are borrowed from [UNISURF](https://github.com/autonomousvision/unisurf), [NeuS](https://github.com/Totoro97/NeuS) and [NSVF](https://github.com/facebookresearch/NSVF). Thanks for these great projects.
