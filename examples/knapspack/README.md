# PSO-JAX

An accelerated Particle Swarm Optimization tool that uses [JAX](https://github.com/google/jax) key component. Here are the main benefits:

* Runs on Python.
* Supports multi-core CPUs, GPUs, and TPUs.
* Compatible with Colab environment.

## Getting Started

### Collab example
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BHI-Research/PSO-JAX/blob/master/examples/knapspack/PSO-JAX-knapspack-example.ipynb)
 

### Local example

In the case of Ubuntu 18.04

```
$ virtualenv --system-site-packages -p python3 ./venv-pso-jax
$ source venv-pso-jax/bin/activate
(venv-pso-jax) $ pip install --upgrade jax jaxlib # CPU-only version
(venv-pso-jax) $ pip install pyswarms
(venv-pso-jax) $ pip install --upgrade tensorflow
(venv-pso-jax) $ python PSO-JAX-knapspack-example.py

```

## Usage

Please cite the following paper:

```
@inproceedings{ermantraut2020resolucion,
  title={Resolución del problema de la mochila mediante la metaheurística PSO acelerada con JAX},
  author={Ermantraut, Joel and Crisol, Tomas and Díaz, Ariel and Balmaceda, Leandro and Rostagno, Adrián and Aggio, Santiago and Blanco, Anibal M and Iparraguirre, Javier},
  booktitle={Simposio Argentino de Inform{\'a}tica Industrial e Investigaci{\'o}n Operativa (SIIIO 2030)-JAIIO 49 (Buenos Aires)},
  year={2020}
}
```

## References

* [BHI Research](https://bhi-research.github.io/)
* [JAX](https://github.com/google/jax)
* [PySwarms](https://pyswarms.readthedocs.io/en/latest/) is used in the examples to compare results.
