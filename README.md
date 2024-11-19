# Shimming Tutorial

This repository contains supplementary code for
Tutorial: Calibration refinement in quantum annealing
https://doi.org/10.3389/fcomp.2023.1238988

Install the dependencies with
```bash
# Python version 3.13
pip install -r requirements.txt
```

Code examples demonstrate orbits for selected models (executed locally),
calculation of embeddings (exectued locally),
and iterative shimming results for several models (executed by default on
Leap-hosted Quantum Processing Units).

Each example has configurable parameters. See the main() function
documentation in each module for a description of parameters. This includes
the flexbility to modify the model, target processor, or embedding
heuristic for increased performance. To reduce the total number
of iterations (programmings, thence QPU access time) or learning rate(s).

## Orbit examples (run locally)

For generation of Figure 4
```
python -m example0_1_orbits
```

For generation of Figure 9
```
python -m example2_1_frustrated_loop_orbits
```

For generation of Figure 11
```
python -m example2_3_buckyball_orbits
```

For generation of orbits related to Figures 13-16
```
python -m example3_1_tafm_get_orbits
```

## Ferromagnetic loop example

Requires 300 sequential QPU-job submissions.
```
python -m example1_1_fm_loop_balancing
```

Requires 300 sequential QPU-job submissions.
```
python -m example1_2_fm_loop_correlations
```

## Frustrated loop example

Requires 300 sequential QPU-job submissions.
```
python -m example2_2_frustrated_loop_anneal
```

## Triangular antiferromagnet example

Generation of embeddings may require several minutes.
Data is collected in 3 blocks each requiring 800 sequential QPU-job submissions.
One figure is presented per block (interrupting data collection until the
figure is closed); along with one summary figure.
```
python -m example3_2_tafm_forward_anneal
```
