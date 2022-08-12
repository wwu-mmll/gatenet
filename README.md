# GateNet
This is the official implementation of the [GateNet paper](https://www.nature.com/ncomms/).

## Tutorials
There are two tutorials which can serve as a starting point for your own analysis.

1. [Tutorial](tutorials/flowcap_data.ipynb): Model training and model prediction (+ cross validation) using the [ND dataset]() published (incl. manual gating) during FlowCAP I

2. [Tutorial](tutorials/custom_data.ipynb): Creation of training data (i.e. manual gating) for custom data using [FlowCytometryTools](https://github.com/eyurtsev/FlowCytometryTools).
This way, manual gating, model training and prediction with custom data is all possible within one Jupyter notebook!

## Why GateNet?
GateNet is a neural network architecture which is specifically designed for automated flow cytometry gating.

### Introduction

Flow cytometry (FC) is an analytical technique which is used to **identify cell types**. 
Inputted with a sample (e.g. blood) containing cells, it **sequentially measures** each cells individual light scatter and fluorescence emission properties.
The cell identification is typically done manually based on 2D scatter plots of the resulting measurements of all cells in the probe.
In manual gating, the scatter points (i.e. cell measurements) are partitioned (’gated’) upon visual inspection.
### Manual Gating
Here, traditional manual gating (of leukocytes i.e. population in the center) of three samples is shown:

![manual gating](data/manual_gating.gif)

As shown, gates have to be set individually for each sample which is both **time consuming and subjectiv**.
Nonetheless, it is inevitable since the distribution of scatter points varies due to measurement variance between samples (’batch effect’).

### Automated Gating (with GateNet)
Gating can be automated using a GateNet which was trained with gated samples.

As shown in the [Tutorial](tutorials/custom_data.ipynb) GateNet can be trained with only 5 training samples.
Since GateNet is implemented in PyTorch, training and prediction can be GPU-accelerated such that automated gating only takes seconds.
These are the **results of the automated gating** (with only 5 training samples) on 15 unseen samples:

![results](data/autogates.png)

### Method

GateNets key feature is to take into account the context of measurements alongside a single cell/event measurement. 
So, to predict the class/type of a single cell the neural network is feed with the single event measurement + (ca. 1000) event measurements of the same sample.  

This allows it to **autonomously correct for batch effects** as it "sees the scatter plot" not just the single scatter point!

![gatenet](data/gatenet.png)

## Installation

### 0. Install Anaconda (if you haven't already)
First, [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) should be installed 

### 1. Create new conda environment
```
conda create -n gatenet python=3.9
conda activate gatenet
```
### 2. Install dependencies
```
conda install -c conda-forge mamba
mamba install -c fastchan fastai anaconda
mamba install -c bioconda fcsparser
mamba install -c conda-forge pyarrow
mamba install -c anaconda seaborn
```

### 3. Install gatenet
```
pip install git+https://github.com/wwu-mmll/gatenet@main
```
