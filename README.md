# shkuratov-torch

> **Status: work in progress.** The API is unstable and the model currently supports only single-component surfaces.

A PyTorch implementation of the Shkuratov model for the spectral albedo of particulate surfaces:

> Shkuratov, Y., Starukhina, L., Hoffmann, H., & Arnold, G. (1999). A model of spectral albedo of particulate surfaces: Implications for optical properties of the Moon. *Icarus*, 137(2), 235–246. https://doi.org/10.1006/icar.1998.6035

A copy of the paper is in [`literature/shkuratov.pdf`](literature/shkuratov.pdf).

This code aims to provide a version of the Shkuratov model that is:

- **Differentiable.** Gradients can be backpropagated through it to optical constants, grain size, and porosity.
- **GPU-friendly.** Computations are vectorised over wavelength, and eventually batches of compositions.
- **Built for generation in bulk** The end goal is fast generation of large libraries of synthetic reflectance spectra.

## Current capabilities

- **Single-component surfaces.** `ShkuratovModel.simulate_one_component` computes albedo for a surface made of one material.
- **Optical constant interpolation.** `n(λ)` and `k(λ)` are linearly interpolated from tabulated values onto the requested wavelengths. Wavelengths outside the tabulated range are clamped to the nearest boundary value.
- **Reflectance coefficients by Gauss–Legendre quadrature.** The backward and forward reflectance coefficients `R_b` and `R_f` are computed by integrating the Fresnel coefficient with 16-point Gauss–Legendre quadrature.

## Installation

There is no package yet. Install the dependencies and run from `src/`:

```bash
pip install torch numpy matplotlib
```

## Usage

```python
import torch
from shkuratov import Material, ShkuratovModel

# Example tabulated optical constants: row 0 = values, row 1 = wavelengths (ascending, no duplicates)
wl = torch.linspace(0.5, 5.0, 500)
n = torch.stack([1.30 + 0.015 * torch.exp(-0.5 * ((wl - 2.0) / 0.18) ** 2), wl])
k = torch.stack([0.001 + 0.08 * torch.exp(-0.5 * ((wl - 2.0) / 0.12) ** 2), wl])

material = Material(n, k)
model = ShkuratovModel()

lams = torch.arange(0.4, 5.6, 0.1)  # requested wavelengths
A = model.simulate_one_component(material, size=5.0, porosity=0.7, lams=lams)  # shape (L,)
```

The `size` argument is the optical path length `S` in the paper and must be in the same units as the wavelengths. The `porosity` argument must lie in [0, 1] and is converted internally to `q = 1 − porosity`, the volume fraction filled by particles (Eq. 10).

## Roadmap

### Model features

- [ ] **Multicomponent surfaces** *(highest priority)*: intimate mixtures of several materials, with per-component fractions and grain sizes (`ShkuratovModel.simulate`).
- [ ] **Batched generation**: evaluate many surfaces (materials × sizes × packing densities) in a single forward pass.
- [ ] **Published approximations for the reflectance coefficients**: an option to use the closed-form empirical approximations of Eq. 8 (`R_b ≈ (0.28n − 0.20)R_e`, `R_e ≈ r_0 + 0.05`, `R_i ≈ 1.04 − 1/n²`) instead of quadrature.

### Diagnostics

- [ ] **Wall-clock benchmarks** for the forward pass on CPU and GPU.
- [ ] **Accuracy/time comparison** between the published reflectance-coefficient approximations and Gauss–Legendre quadrature at different numbers of nodes.

## Citation

If you use this code, please cite the original model:

```bibtex
@article{shkuratov1999,
  title   = {A Model of Spectral Albedo of Particulate Surfaces: Implications for Optical Properties of the Moon},
  author  = {Shkuratov, Yurij and Starukhina, Larissa and Hoffmann, Harald and Arnold, Gabriele},
  journal = {Icarus},
  volume  = {137},
  number  = {2},
  pages   = {235--246},
  year    = {1999},
  doi     = {10.1006/icar.1998.6035}
}
```
