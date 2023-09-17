# Minimal Bayesian Flow Networks

![animation](notebooks/bayesian_flow_with_trajectories.gif)

A minimal reproduction of [Bayesian Flow Networks](https://arxiv.org/abs/2308.07037) over continuous data for the purposes of clarity and understanding. The notebooks in this repo reproduce the figures in Section 4 of the paper and apply BFNs to model 1-dimensional and 2-dimensional points.

## Installation
Install the required packages.
```
pip install -r requirements.txt
```
Install `bfn` as a package.
```
pip install -e .
```

## Notebooks
- `bayesian_update_function.ipynb`: reproduces Figure 2
- `bayesian_update_distribution.ipynb`: reproduces Figure 3
- `trajectories.ipynb`: reproduces Figure 4
- `accuracy_schedule.ipynb`: visualizes the accuracy schedule
- `1d_point.ipynb`: trains and samples from a BFN modeling a scalar
- `2d_point.ipynb`: trains and samples from a BFN modeling a two-dimensional point

## Contributing
If you find any bugs, please let me know at mbchang [at] berkeley [dot] edu.

## License
MIT