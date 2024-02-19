# Pseudo-spectral solver for 2D-HIT

This project contains a pseudo-spectral solver for forced HIT (homogeneous isotropic turbulence).

We consider the NS equations in vorticity formulation:

$$\frac{\partial \omega}{\partial t} + \mathcal{J}(\omega, \psi) = \nu \nabla^2 \omega + \mu(f - \omega),$$

$$\nabla^2 \psi= \omega,$$

where $J$ is the advection operator

$$
    \mathcal{J}(\omega, \psi) = \frac{\partial \psi}{\partial x} \frac{\partial \omega}{\partial y} - \frac{\partial \psi}{\partial y} \frac{\partial \omega}{\partial x}.
$$

Let $\hat{u}_{\boldsymbol{k}}$ denote the Fourier coefficient of wave number vector $\boldsymbol{k}$ for the Fourier transform of the scalar function $u(x,y)$. So 

$$u(x,y) = \sum_{\boldsymbol{k} \in \mathbb{Z}^2} \hat{u}_{\boldsymbol{k}} e^{i(k_1 x + k_2 y)} .$$

## Requirments

This project requires PyTorch. It is developed to use GPU acceleration, but it should also work on "CPU only" devices (not tested). You may also need Git LFS (Large File Storage) to download the large files in the repository, as it was used to upload them.

## Workflow for setting up a subgrid parametrization

### High fidelity reference simulation

Both the CNN base-parametrization and the tau-orthogonal method require training data. For the CNN this training data consists of {input: low fidelity fields;  output: SGS field}. The tau orthogonal method needs the trajectory of the quanities of interest in a high fidelity simulation (its targets to track).

The training datasets can be created using "compute_reference_torch.py". 
Which implements a pseudo-spectral AB/BDI2 scheme. 
For details, please see "more_on_inputs.md". 

The CNNs are typically trained on a data set with 2000 input -> output fields. These can be obtained from a 2000-day simulation. Such a simulation takes up to 3 hours with GPU acceleration. After which it has created a file with training data for the CNNs and a file with reference data for the tau-orthogonal method.


https://github.com/rik-stra/minimal_code_for_tau_orthogonal_subgrid_CNNs/assets/124180091/066299e9-054e-49c5-ad7f-c71e32c67023



