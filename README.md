# ADOptDiff - An Affinity Driven R-chain Diffusion Model for Lead Compounds Optimization
ADOptDiff is a diffusion model that automatically modifies chemical scaffolds to enhance the binding affinity between molecules and their targets.

## Install conda environment via conda yaml file
```bash
conda env create -f environment.yaml
```
## Datasets
Coming soon
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
Run the evaluation script after the sampling process is complete.
```bash
bash evaluate.sh
```
