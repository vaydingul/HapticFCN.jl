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

## V.0.1.1
In this version, mostly, the version v.0.1 is improved. The results and discussion about this version can be found in this [Excel Sheet](https://docs.google.com/spreadsheets/d/1KO6d-lZPePWM3OUPCAwU7RIf10dt26h3WW39059Uwas/edit?usp=sharing) and this [Word Document](https://docs.google.com/document/d/1QzXVCBX1liEPLB9E1T35SLpVcXVzFFjDJmBKs3h2mW4/edit?usp=sharing). In this version, the following updates were applied to the model:

* Custom implementation of *Local Response Normalization*
* L2 Regularization implementation
* Fix on the *haptic* data normalization



## Current Situation

Currently, the project is not successfully replicated. The main reason is considered as the level of replicability of the main paper.
Future work will continue on the latest branch (v.0.2.2). 


## Project Deliverables


[Data Sheet](https://docs.google.com/spreadsheets/d/1KO6d-lZPePWM3OUPCAwU7RIf10dt26h3WW39059Uwas/edit?usp=sharing)


[Research Log](https://docs.google.com/document/d/1QzXVCBX1liEPLB9E1T35SLpVcXVzFFjDJmBKs3h2mW4/edit?usp=sharing)


[Presentation](https://docs.google.com/presentation/d/1cuKA4-ZxECWkBhPrMd1y_-96mtv0fQAYAy8M5A_3bMM/edit?usp=sharing)


[Technical Report](https://www.overleaf.com/read/kzvsntmwmtgv)
