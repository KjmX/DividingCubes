<p align="center">
    <img src="docs/assets/dividing-cubes-cuda.png">
	<br>
</p>

# Dividing Cubes with <span style="color: #76b900;"> CUDA </span>

A sequentiel and parallel implementation of the [Dividing Cubes algorithm](https://doi.org/10.1118/1.596225) using C++ and CUDA.

## Abstract

3D image reconstruction has the potential to bring major advancements in science and medicine by allowing the visualization of inner living organs in their real states and forms.
Using modern modalities such as Computed Tomography (CT) and Magnetic Resonance Imaging (MRI), serial 2D images are produced and used in 3D reconstruction.
However, the reconstruction process is very slow and very expensive in terms of compute resources due to the massive quantity of data to process resulted from the acquisition task.
Meanwhile, Graphic Processing Unit (GPU), with its tremendous capability of parallel computing, becomes more and more popular in High Performance Computing.
In addition, CUDA, a parallel computing platform and programming model that was invented by NVIDIA, makes GPU programming much easier and faster.

In our research, we focused on the use of this power of parallel computing in order to accelerate the reconstruction process while trying to have the most accurate representation of the reconstructed object.

### Keyword

*3D Medical Imaging*, *Image Reconstruction*, *Dividing Cubes*, *GPU*, *CUDA*.

## Outputs

<span style="color: yellow;">TODO</span>

<p align="center">
    <img src="docs/figures/Page-67-Image-88.png">
	<br>
</p>

<span style="color: yellow;">TODO</span>

<p align="center">
    <img src="docs/figures/Page-68-Image-89.png">
	<br>
</p>

## Modules

### Data Reader Module

<span style="color: yellow;">TODO</span>

<p align="center">
    <img src="docs/figures/Page-65-Image-86.png">
	<br>
</p>

### Squential (Non-Parallel) Module

<span style="color: yellow;">TODO</span>

<p align="center">
    <img src="docs/figures/Page-62-Image-84.png">
	<br>
</p>

### CUDA (Parallel) Module

<span style="color: yellow;">TODO</span>

<p align="center">
    <img src="docs/figures/Page-63-Image-85.png">
	<br>
</p>

### Visualization Module

<span style="color: yellow;">TODO</span>

<p align="center">
    <img src="docs/figures/Page-66-Image-87.png">
	<br>
</p>

## Notes

- Running this code requires a CUDA capable GPU (compute capability 2.2)
- The datasets used in this project have all been obtained from [here](http://www.gris.uni-tuebingen.de/edu/areas/scivis/volren/datasets/datasets.html)

<span style="color: yellow;">TODO</span>

## Authors

- Mohamed Tahar KEDJOUR [@kjmx](https://github.com/KjmX)

- Anis LOUNIS [@anixpasbesoin](https://github.com/AnixPasBesoin)