# COMP541 Project
Koc University 2020/2021 Fall COMP541 Project

This GitHub project includes all the related files with COMP541 Deep Learning Course project.

In this project, the aim is to replicate the previous study called:

> Deep learning for surface material classification using haptic and visual information

## Development
The project development is being held using the sophisticated deep learning framework called [Knet](https://github.com/denizyuret/Knet.jl)

## Baseline Model
The baseline model can be reached from this [notebook](https://github.com/vaydingul/COMP541_Project/blob/main/baseline_model/baseline.ipynb).

The baseline model includes:
* Data preprocessing
* Simple network architecture
* Short training
* Visualization of the results
* Metric calculation

## V.0.1
The baseline model is improved, and the one of the neural network, that is studied in the project paper, is constructed, namely HapticNet. The demonstration of the HapticNet can be found in this [notebook](https://github.com/vaydingul/COMP541_Project/blob/main/v.0.1/HapticNet.ipynb).

The V.0.1 includes:
* Improved data processing
* New loss calculation scheme
* New accuracy calculation scheme based on *max-voting* procedure
