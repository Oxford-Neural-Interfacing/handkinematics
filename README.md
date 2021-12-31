# Kinematic features derived from temporal dynamics allow accurate reconstruction of hand movements
## Conor Keogh & James J FitzGerald
### Oxford Neural Interfacing, Nuffield Department of Surgical Sciences, University of Oxford

## Abstract
The human hand is a uniquely complex effector. The ability to describe hand kinematics with a small number of features suggests that complex hand movements are composed of combinations of simpler movements. This would greatly simplify the neural control of hand movements. If such movement primitives exist, a dimensionality reduction approach designed to exploit these features should outperform existing methods. We developed a deep neural network to capture the temporal dynamics of movements and demonstrate that the features learned allow accurate representation of functional hand movements using lower-dimensional representations than previously reported. We show that these temporal features are highly conserved across individuals and can interpolate previously unseen movements, indicating that they capture the intrinsic structure of hand movements. These results indicate that functional hand movements are defined by a low-dimensional basis set of movement primitives with important temporal dynamics and that these features are common across individuals.

## Repository contents
This repository contains analysis code for assessment of dimensionality reduction techniques for representation of hand kinematics using features with temporal dynamics.

The ```analysis``` folder contains all Python scripts for training and evaluating dimensionality reduction models, including:
- Reconstruction of kinematic data
- Classification of reconstructed movements
- Interpolation of previously unseen movements

These analyses are applied using:
- Models trained on individuals' own data (individualized models)
- Models trained on all other participants' data (transferred models)
- Models trained on all data (grand-average models)

Separate scripts are included for analysis of processed results and plot generation.

Data and results folders are not included with the repository (see below).

## Contact
For enquiries related to this project, please contact:

Conor Keogh: <conor.keogh@nds.ox.ac.uk>  
James FitzGerald: <james.fitzgerald@nds.ox.ac.uk>

## Data availability
Raw kinematic data is not included in this repository due to large file sizes. This data will be transferred by the authors on request.

## Acknowledgements
This work was further supported by the National Institute for Health Research (NIHR) through the NIHR Oxford Biomedical Research Centre and the UK Engineering and Physical Sciences Research Council (EPSRC) via the University of Oxford Clarendon Fund. The views expressed are those of the authors and not necessarily those of the funding bodies.
