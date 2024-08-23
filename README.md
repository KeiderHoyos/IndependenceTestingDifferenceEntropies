# DiME: Maximizing Mutual Information via Difference of Matrix-based Entropies

In this repository, we perform experiments for independence testing using a new measurement to capture dependece between random variables. The proposed measure is based on the matrix-based (kernel-based) mutual information, with learnable kernels.  

The proposed method is compared to state-of-the-art methods for independence testing using the Hilbert-Schmidt Independence criterion (HSIC). 

The baselines are from the following repository: [HSIC-LK/](https://github.com/renyixin666/HSIC-LK/) under MIT License (see [LICENSE.md](https://github.com/renyixin666/HSIC-LK/blob/main/LICENSE)).

## Requirements

The requirements for the packages are:
- `python 3.7+`
  - `numpy`
  - `scipy`
  - `cupy`
  - `pytorch`

## HSIC with learnable kernels

**Statistical Independence Test.** Given the tensor $X$ of shape $(n, d_x)$ and $Y$ of shape $(n, d_y)$, our test returns $0$ if the samples $X$ and $Y$ are independence, and $1$ otherwise.
For details, check out the [demo.ipynb](./demo.ipynb)

## Bibtex

```
@article{skean2023dime,
  title={Dime: Maximizing mutual information by a difference of matrix-based entropies},
  author={Skean, Oscar and Hoyos-Osorio, Jhoan Keider and Brockmeier, Austin J and Giraldo-Sanchez, Luis Gonzalo},
  journal={arXiv preprint arXiv:2301.08164},
  year={2023}
}
```




