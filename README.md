# ETDRK4

This repository contains a Python implementation of ETDRK4 (Exponential Time Differencing fourth-order RungeKutta) [[1](#citations)] method for solving stiff nonlinear PDEs. 

In particular both PyTorch and Numpy implementations for the Kuramoto-Sivashinsky equation are provided, ported from the MATLAB code proposed by Aly-Khan Kassam and Lloyd N. Trefethen in ["Fourth-Order Time-Stepping for Stiff PDEs"](https://doi.org/10.1137/S1064827502410633).

<p align="center">
  <img src="figure.png" />
</p>

## Citations

    [1] @article{doi:10.1137/S1064827502410633,
            author = {Kassam, Aly-Khan and Trefethen, Lloyd N.},
            title = {Fourth-Order Time-Stepping for Stiff PDEs},
            journal = {SIAM Journal on Scientific Computing},
            volume = {26},
            number = {4},
            pages = {1214-1233},
            year = {2005},
            doi = {10.1137/S1064827502410633},
            URL = {https://doi.org/10.1137/S1064827502410633},
            eprint = {https://doi.org/10.1137/S1064827502410633}
        }

