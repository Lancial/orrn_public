## ORRN: An ODE-based Recursive Registration Network for Deformable Respiratory Motion Estimation with Lung 4DCT Images
by Xiao Liang, Shan Lin, Fei Liu, Dimitri Schreiber, and Michael Yip. Accepted by IEEE Transaction on Biomedical Engineering (TBME)


## Installation

Install environment with docker: 
```
cd docker
docker build -t orrn .
cd ..
```
Last time we tested successfully with python 3.7.11, pytorch 1.10.0, and CUDA 11.1

## Data preparation

Our model is trained on [4D-Lung](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=21267414) and [SPARE](https://image-x.sydney.edu.au/spare-challenge/) dataset. We provide some dataloader in `data.py` file. However, to use them, one has to convert training data to the following format:

```
<dataset_name>:
    <patient1>:
        <trial1>:
            images:
                0.mha
                1.mha
                ...
                9.mha
        <trial2>:
    <patient2>:
```
and modifies the `SEQ_DATA_PATH` variable in `data.py`. For details, please refer to the paper.


One can use their own data for training. The training code expects data with shape `(B, 2, H, W, D)` for pair-wise registration, and `(B, 10, H, W, D)` for group-wise registration.

## Training
Currently, single level pair-wise and group-wise registration code is released, multi-level pair-wise registration code is coming soon.
To train the ORRN model:

```
python -W ignore train.py
```

## Citing our work
If you find our work useful, please consider citing
```
@ARTICLE{10144816,
    author={Liang, Xiao and Lin, Shan and Liu, Fei and Schreiber, Dimitri and Yip, Michael},
    journal={IEEE Transactions on Biomedical Engineering}, 
    title={ORRN: An ODE-based Recursive Registration Network for Deformable Respiratory Motion Estimation With Lung 4DCT Images}, 
    year={2023},
    volume={},
    number={},
    pages={1-12},
    doi={10.1109/TBME.2023.3280463}
}

```

## Thanks
Our code is based on [VoxelMorph](https://github.com/voxelmorph/voxelmorph) and [RRN](https://github.com/Novestars/Recursive_Refinement_Network)


