# Non-Graph Data Clustering via O(n) Bipartite Graph Convolution


This repository is our implementation of 

[Hongyuan Zhang, Jiankun Shi, Rui Zhang, and Xuelong Li,  "Non-Graph Data Clustering via O(n) Bipartite Graph Convolution," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, DOI:10.1109/TPAMI.2022.3231470, 2022](https://ieeexplore.ieee.org/document/9996549).

AnchorGAE attempts to accelarate the unsupervised GNN (e.g., [AdaGAE](https://github.com/hyzhang98/AdaGAE)), which could be used to promote the clustering performance, via the classical  trick of anchors / landmarks. It leads to a Siamese architecture and a specific graph convolution operation. 

It should be emphasized that AnchorGAE is designed for the clustering on non-graph data, where all data points are only represented by $d$-dimension vectors and the graph is not provided as priori. It could be regarded as an GNN extension of scalable graph clustering. 

If you have issues, please email:

hyzhang98@gmail.com or henusjk@163.com.

## How to Run AnchorGAE

To run the experiment, the name of dataset and parameters need be required. The required configuration is explained at the end.

### Name of Dataset

There are six datasets are provided.

```Parameter: {datasetName}```

```
--datasetName=usps_all

--datasetName=segment_uni

--datasetName=mnist_all

--datasetName=Isolet

--datasetName=fashionMNIST_full

--datasetName=mnist_test
```

### Parameters

There are three hyperparameters that need to be set.

```Parameter1: {AnchorNum, type:int, help='Initialize the number of anchors.'}```

```Parameter2: {k0, type:int, Initialize the k-sparsity.}```

```Parameter3: {increase_k, type:int, help='Initialize the size of the self-increasing sparsity.'}```

### Run

There is an example running on USPS dataset.

```
python Main.py --datasetName=usps_all --AnchorNum=400 --increase_k=6 --k0=3
```

### Requirements 

- pytorch 1.3.1
- torchvision 0.4.2
- munkres 1.0.12
- scipy 1.3.1
- scikit-learn 0.21.3
- numpy 1.16.5

## Citation
```
@article{AnchorGAE,
  author={Zhang, Hongyuan and Shi, Jiankun and Zhang, Rui and Li, Xuelong},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Non-Graph Data Clustering via O(n) Bipartite Graph Convolution}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2022.3231470}
}
```
