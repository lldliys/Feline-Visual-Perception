# Computational Modeling of the Feline Visual Perception

## Overview
This project implements a multi-stage image processing pipeline to reconstruct the subjective visual experience (**Umwelt**) of the domestic cat (*Felis catus*). The simulation is grounded in biological constraints, mapping physiological data of the feline visual system onto a robust computational model.

## Computational Modules
1. **Dichromatic Color Transformation**: I implemented a color-space conversion matrix derived from the spectral sensitivity of feline cone opsins (peaking at ~447nm and ~555nm). This simulates the protanopic-like vision of cats, who lack long-wavelength (red) cones.
2. **Spatio-Temporal Resolution Modeling**: 
    * **Spatial**: Implemented a radial blur mask that simulates the trade-off between foveal clarity and peripheral sensitivity.
    * **Temporal**: Since cats have a higher Flicker Fusion Frequency (FFF) than humans (approx. 70-80 Hz vs. 60 Hz), I engineered a frame-interleaving algorithm to simulate the potential for "motion blur" or "frame-flicker" perception under lower-frequency lighting.
3. **Spherical FOV Distortion**: Developed a fisheye-correction algorithm (using camera matrix `K` and distortion coefficients `D`) to emulate the ~200° panoramic field of view, far exceeding the human ~180° limit.

## Engineering Stack
* **Language**: Python 3.11
* **Libraries**: OpenCV (Computer Vision), NumPy (Linear Algebra/Matrix Operations)

## License
This project is for academic purposes as part of NEU 172L coursework at UC Berkeley.
