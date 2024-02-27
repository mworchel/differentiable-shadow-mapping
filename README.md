<p align="center">

  <h1 align="center"><a href="https://mworchel.github.io/differentiable-shadow-mapping">Differentiable Shadow Mapping for Efficient Inverse Graphics</a></h1>

  <a href="https://mworchel.github.io/differentiable-shadow-mapping">
    <img src="docs/static/images/teaser.jpg" alt="Logo" width="100%">
  </a>

  <p align="center">
    <i>IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2023</i>
    <br />
    <a href="https://github.com/mworchel"><strong>Markus Worchel</strong></a>
    ·
    <a href="https://www.cg.tu-berlin.de/team/prof-dr-marc-alexa/"><strong>Marc Alexa</strong></a>
  </p>
</p>

## About

This repository contains the official implementation of the paper "Differentiable Shadow Mapping for Efficient Inverse Graphics", which presents a technique for efficient shadow rendering in the context of differentiable rasterization.
The implementation is based on the differentiable rasterization primitives from [nvdiffrast](https://github.com/NVlabs/nvdiffrast).

Similar to nvdiffrast, we provide a set of flexible low-level primitives for differentiable shadow mapping. These primitives are meant to be used when implementing a custom differentiable renderer with shadow mapping. They are contained in the module `diffshadow.shadow`: 
- `render_shadow_map`: Render the scene(s) from the point-of-view of the light(s) and generate a shadow map with the necessary information (e.g. the depth for standard shadow mapping)
- `filter_shadow_map`: Apply the anti-aliasing operation of nvdiffrast and convolve the shadow map(s) with a filter kernel
- `compute_visibility`: From the (filtered) shadow map(s) compute the visibility for an array of points in the scene

In addition, we provide a simple renderer that implements a high-level interface to our shadow mapping primitives. A variant of this renderer has been used to generate the results in the paper. The package `diffshadow.simple_renderer` contains the renderer and utilities for scene generation and geometry processing.

## Getting Started 

Setup the environment and install basic requirements using conda

```bash
conda env create -f environment.yml
conda activate diffshadow
```

In the directory where you have cloned this repository to, install our package by running

```bash
pip install .
```

Now, you can run any of the notebooks provided in this repository:
- `1_getting_started.ipynb`: Introduction to the simple renderer and its scene representation with a basic shadow-driven pose optimization
- `2_shadow_art.ipynb`: Simple shadow art experiments
- `3_minimal_plane.ipynb`: Reproduction of Figure 5 in the paper (necessity for applying anti-aliasing to the shadow map)
- `4_low_level_primitives.ipynb`: Examples for how to use the low-level shadow mapping primitives



Alternatively, if you want to use the implementation without running any notebooks, you might install our package by simply running

```bash
pip install git+https://github.com/mworchel/differentiable-shadow-mapping
```

## Common Issues and Pitfalls

### "Rendered images are blank or I don't see any shadows"

Shadow mapping relies on rendering from the perspective of the light. The frustum of these light cameras must contain the scene to generate useable data. A common pitfall is the incorrect configuration of near and far planes, such that the scene or parts of it are culled when rendering from the light perspective. Make sure the values of `near`, `far`, and `distance` (last only for directional lights) are properly adjusted to your scene and light positions.

## Citation

If you find this code or our method useful for your academic research, please cite our paper

```bibtex
@InProceedings{worchel:2023:diff_shadow_mapping,
      author    = {Worchel, Markus and Alexa, Marc},
      title     = {Differentiable Shadow Mapping for Efficient Inverse Graphics},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      month     = {June},
      year      = {2023},
      pages     = {142-153}
  }
```