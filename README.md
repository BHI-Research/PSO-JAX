# PSO-JAX

An accelerated Particle Swarm Optimization tool that uses [JAX](https://github.com/google/jax) key component. Here are the main benefits:

* Runs on Python.
* Supports multi-core CPUs, GPUs, and TPUs.
* Compatible with Colab environment.

## Getting Started

### Collab example
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BHI-Research/PSO-JAX/notebooks/PSO-JAX-knapspack-example.ipynb)
 

### Local example

In the case of Ubuntu 18.04

```
$ virtualenv --system-site-packages -p python3 ./venv-pso-jax
$ source venv-pso-jax/bin/activate
(venv-pso-jax) $ pip install --upgrade jax jaxlib  # CPU-only version
(venv-pso-jax) $ pip install pyswarms
(venv-pso-jax) $ python PSO-JAX-knapspack-example.py 

```

## References

* [BHI Research](https://bhi-research.github.io/)
* [JAX](https://github.com/google/jax)
* [PySwarms](https://pyswarms.readthedocs.io/en/latest/) is used in the examples to compare results.
