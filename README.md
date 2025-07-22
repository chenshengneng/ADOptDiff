# ADOptDiff - An Affinity Driven R-chain Diffusion Model for Lead Compounds Optimization
ADOptDiff is a diffusion model that automatically modifies chemical scaffolds to enhance the binding affinity between molecules and their targets.
<p align='center'>
<img src=".ADOptDiff.pdf" alt="architecture"/> 
</p>

## Install conda environment via conda yaml file
```bash
conda env create -f environment.yaml
```
## Datasets

## Training
```bash
python train.py --config configs/ADOptDiff.yml
```
## Sampling
Modify each sample in the test set 100 times.
```bash
bash sample.sh
```
## Evaluation
```bash
bash evaluate.sh
```
