# GateNet
This is the official implementation of the [GateNet paper](https://www.nature.com/ncomms/). 
## Introduction
GateNet is a neural network architecture which is specifically designed for automated flow cytometry gating.

Flow cytometry (FC) is an analytical technique which is used to **identify cell types**. 
Inputted with a sample (e.g. blood) containing cells, it **sequentially measures** each cells individual light scatter and fluorescence emission properties.
The cell identification is typically done manually based on 2D scatter plots of the resulting measurements of all cells in the probe.
In manual gating, the scatter points (i.e. cell measurements) are partitioned (’gated’) upon visual inspection.
### Manual Gating
Here, traditional manual gating (of leukocytes i.e. population in the center) of three probes is shown:

![manual gating](data/manual_gating.gif)

As shown, gates have to be set individually for each sample which is both **time consuming and subjectiv**.
Nonetheless, it is inevitable since the distribution of scatter points varies due to measurement variance between samples (’batch effect’).

### Automated Gating (with GateNet)
Gating can be automated using a GateNet which was trained with gated samples.

As shown in examples/custom_tutorial.ipynb GateNet can be trained with only 5 training samples.
Since GateNet is implemented in PyTorch, GPU-acceleration during training and prediction is possible such that automated gating only takes seconds.


![results](data/autogates.png)

GateNets key feature is to take into account the context of measurements alongside the single event measurement for its prediction.

![gatenet](data/gatenet.png)


