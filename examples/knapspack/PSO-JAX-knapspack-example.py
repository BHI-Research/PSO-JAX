# -*- coding: utf-8 -*-


def init_knapsack():
    global pv_file
    global bs_file
    global rs_file

    pv_file = np.loadtxt('pv.csv', delimiter=',')
    rs_file = np.loadtxt('rs.csv', delimiter=',')
    bs_file = np.loadtxt('bs.csv', delimiter=',')


def knapsack_res(X):
    weight = np.sum(pv_file, axis=0)

    PR = np.matmul(pv_file, X.transpose()).transpose()

    G = np.matmul(rs_file, X.transpose()).transpose()
    # Transpose to reach (k,j),(j,i)->(k,i)

    VIO = np.add(G, -1 * bs_file)

    VIO = np.where(VIO < 0.0, 0.0, VIO)

    PNLTY = np.sum(VIO, axis=1)

    FI = np.add(PR, -1 * (weight * PNLTY))

    return FI, VIO


def knapsack(X):
    FI, VIO = knapsack_res(X)

    return -FI


def activation(x, function=1):
    if (function == 1):  # Sigmoid
        return 1 / (1 + np.exp(-x))
    elif (function == 2):  # Softmax
        expo = np.exp(x)
        expo_sum = np.sum(np.exp(x))
        return expo / expo_sum
    elif (function == 3):  # ReLu
        return np.maximum(0, x)


def toDiscrete(x):
    return np.where(x < 0.5, 0, 1)


def convert_to_discrete(x):
    x_sigmoid = 1 / (1 + np.exp(-x))
    discrete_position = np.where(x_sigmoid < 0.5, 0, 1)
    return discrete_position


def calculate_velocity(w, particles_velocity, c1, c2, r1, r2, best_particle_position,
                       particles_position, best_global_position):
    inertia = w * particles_velocity
    best_particle_pos_component = r1 * (best_particle_position - particles_position)
    best_global_pos_component = r2 * (best_global_position - particles_position)

    new_velocity = inertia + c1 * best_particle_pos_component + c2 * best_global_pos_component
    return new_velocity


def calculate_best_position(objective_values, best_particle_cost, particles_position, best_particle_position, particles,
                            dimensions):
    bests = np.less(objective_values, best_particle_cost)
    reshape = np.reshape(bests, np.array([particles, 1]))
    bests_reshape = np.broadcast_to(reshape, np.array([particles, dimensions]))
    pos = np.where(bests_reshape, particles_position, best_particle_position)
    return pos


def runDiscretePSO_np(user_options, algorithm_options):
    particles = algorithm_options['particles']
    dimensions = algorithm_options['dimensions']
    objective = algorithm_options['objective']
    # For each particle, initialize position and velocity
    particles_position = np.random.uniform(-1, 1, (particles, dimensions))
    particles_velocity = np.random.uniform(-1, 1, (particles, dimensions))

    particles_position = toDiscrete(activation(particles_velocity))

    best_global = None  # Best swarm cost
    best_global_position = np.empty((particles, dimensions))  # Best swarm position
    best_particle_position = particles_position
    best_particle_cost = objective(best_particle_position)

    for k in range(0, algorithm_options['iterations']):
        objective_values = objective(best_particle_position)
        best_index = np.argmin(objective_values)
        best_value = objective_values[best_index]

        # particles x dimensions
        best_particle_position = calculate_best_position(objective_values, best_particle_cost, particles_position,
                                                         best_particle_position, particles, dimensions)

        if best_global is None or best_value < best_global:
            # Update best swarm cost and position
            best_global = best_value
            best_global_position = particles_position[best_index]

        r1 = np.random.uniform(0, 1, (particles, dimensions))
        r2 = np.random.uniform(0, 1, (particles, dimensions))

        particles_velocity = calculate_velocity(user_options['w'], particles_velocity, user_options['c1'],
                                                user_options['c2'], r1, r2, best_particle_position,
                                                particles_position, best_global_position)

        particles_position = toDiscrete(activation(particles_position + particles_velocity))

        best_particle_position = particles_position

    return best_global, best_global_position


"""# PSO/Knapsack with JAX.NUMPY"""

from datetime import datetime

import jax.numpy as npj
import numpy as np
from jax import random, jit


@jit
def knapsack_res(X):
    weight = npj.sum(pv_file, axis=0)

    PR = npj.matmul(pv_file, X.transpose()).transpose()

    G = npj.matmul(rs_file, X.transpose()).transpose()
    # Transpose to reach (k,j),(j,i)->(k,i)

    VIO = npj.add(G, -1 * bs_file)

    VIO = npj.where(VIO < 0.0, 0.0, VIO)

    PNLTY = npj.sum(VIO, axis=1)

    FI = npj.add(PR, -1 * (weight * PNLTY))

    return FI, VIO


@jit
def knapsack(X):
    FI, VIO = knapsack_res(X)

    return -FI


@jit
def activation(x, function=1):
    if (function == 1):  # Sigmoid
        return 1 / (1 + npj.exp(-x))
    elif (function == 2):  # Softmax
        expo = npj.exp(x)
        expo_sum = npj.sum(npj.exp(x))
        return expo / expo_sum
    elif (function == 3):  # ReLu
        return npj.maximum(0, x)


@jit
def toDiscrete(x):
    return npj.where(x < 0.5, 0, 1)


@jit
def convert_to_discrete(x):
    x_sigmoid = 1 / (1 + npj.exp(-x))
    discrete_position = npj.where(x_sigmoid < 0.5, 0, 1)
    return discrete_position


@jit
def calculate_velocity(w, particles_velocity, c1, c2, r1, r2, best_particle_position,
                       particles_position, best_global_position):
    inertia = w * particles_velocity
    best_particle_pos_component = r1 * (best_particle_position - particles_position)
    best_global_pos_component = r2 * (best_global_position - particles_position)

    new_velocity = inertia + c1 * best_particle_pos_component + c2 * best_global_pos_component
    return new_velocity


def calculate_best_position(objective_values, best_particle_cost, particles_position, best_particle_position, particles,
                            dimensions):
    bests = npj.less(objective_values, best_particle_cost)
    reshape = npj.reshape(bests, npj.array([particles, 1]))
    bests_reshape = npj.broadcast_to(reshape, npj.array([particles, dimensions]))
    pos = npj.where(bests_reshape, particles_position, best_particle_position)
    return pos


def runDiscretePSO_jax(user_options, algorithm_options):
    particles = algorithm_options['particles']
    dimensions = algorithm_options['dimensions']
    objective = algorithm_options['objective']
    # For each particle, initialize position and velocity
    seed = random.PRNGKey(datetime.now().microsecond)
    particles_position = random.uniform(seed, (particles, dimensions), None, -1, 1)
    seed = random.PRNGKey(datetime.now().microsecond)
    particles_velocity = random.uniform(seed, (particles, dimensions), None, -1, 1)
    # Use of system microseconds as random seed to get different numbers each time

    particles_position = toDiscrete(activation(particles_velocity))

    best_global = None  # Best swarm cost
    best_global_position = npj.empty((particles, dimensions))  # Best swarm position
    best_particle_position = particles_position
    best_particle_cost = objective(best_particle_position)  # obj_fuction(best_particle_position)

    for k in range(0, algorithm_options['iterations']):
        # Don't replace with 'iterations' variable because it is called only once
        objective_values = objective(best_particle_position)  # obj_fuction(particles_position)
        best_index = npj.argmin(objective_values)
        best_value = objective_values[best_index]

        # particles x dimensions
        best_particle_position = calculate_best_position(objective_values, best_particle_cost, particles_position,
                                                         best_particle_position, particles, dimensions)

        if best_global is None or best_value < best_global:
            # Update best swarm cost and position
            best_global = best_value
            best_global_position = particles_position[best_index]

        seed = random.PRNGKey(datetime.now().microsecond)
        r1 = random.uniform(seed, (particles, dimensions), None, 0, 1)
        seed = random.PRNGKey(datetime.now().microsecond)
        r2 = random.uniform(seed, (particles, dimensions), None, 0, 1)

        particles_velocity = calculate_velocity(user_options['w'], particles_velocity, user_options['c1'],
                                                user_options['c2'], r1, r2, best_particle_position,
                                                particles_position, best_global_position)

        particles_position = toDiscrete(activation(particles_position + particles_velocity))

        best_particle_position = particles_position

    return best_global, best_global_position


"""# Compare solutions"""

import time
import pyswarms as ps


# This function generates the instance needed to prove the PSO
def instance_generation(n_items, n_constraints, alpha=0.75):
    rs = np.random.randint(1000, size=(n_constraints, n_items))
    bs = alpha * np.sum(rs, axis=1)
    q = np.random.random_sample((n_items))
    pv = np.sum(rs, axis=0) / n_constraints
    pv = np.add(pv, 500 * q)

    return pv, rs, bs


def print_solution(solX):
    solPru = np.random.randint(2, size=(1, n_items))
    for k in range(0, n_items):
        solPru[0, k] = solX[k]

    print(solX)
    print(solPru)

    [f, vio] = knapsack_res(solPru)
    print(vio)


def ps_bhi_np_solution(n_particles, iterations, dim, ftol, options):
    global solX_NumPy
    user_options = {'c1': options['c1'], 'c2': options['c2'], 'w': options['w']}
    algorithm_options = {'particles': n_particles, 'dimensions': dim,
                         'iterations': iterations, 'objective': knapsack, 'ftol': ftol}

    # Perform optimization
    start = time.time()
    solFO, solX_NumPy = runDiscretePSO_np(user_options, algorithm_options)
    wall_time = time.time() - start

    return solFO, wall_time


def pso_bhi_jax_solution(n_particles, iterations, dimensions, ftol, options):
    global solX_NumPyJAX
    user_options = {'c1': options['c1'], 'c2': options['c2'], 'w': options['w']}
    algorithm_options = {'particles': n_particles, 'dimensions': dimensions,
                         'iterations': iterations, 'objective': knapsack, 'ftol': ftol}

    # Perform optimization
    start = time.time()
    solFO, solX_NumPyJAX = runDiscretePSO_jax(user_options, algorithm_options)
    wall_time = time.time() - start

    return solFO, wall_time


def pyswarm_solution(n_particles, iterations, dimensions, ftol, options):
    global solX_PySwarms
    optimizer = ps.discrete.binary.BinaryPSO(n_particles=n_particles,
                                             dimensions=dimensions,
                                             options=options,
                                             init_pos=None,
                                             velocity_clamp=None,
                                             vh_strategy='unmodified', ftol=ftol)

    # Perform optimization
    start = time.time()
    [solFO, solX_PySwarms] = optimizer.optimize(knapsack, iters=iterations)
    wall_time = time.time() - start

    return solFO, wall_time


def compare_solutions():
    global pv_file
    global bs_file
    global rs_file
    global n_items
    global n_constraints

    n_items = 20
    n_constraints = 5

    [pv_temp, rs_temp, bs_temp] = instance_generation(n_items, n_constraints)
    np.savetxt('pv.csv', pv_temp, delimiter=',')
    np.savetxt('rs.csv', rs_temp, delimiter=',')
    np.savetxt('bs.csv', bs_temp, delimiter=',')

    dimensions = n_items
    iteracions = 100
    n_particles = 15

    average_iters = 5
    ftol = -np.inf

    wall_times = [0, 0, 0]
    solFOs = [0, 0, 0]
    # Item 0: Pyswarms Solution
    # Item 1: PSO-BHI Solution using Pyswarms
    # Item 2: PSO-BHI Solution using JAX

    init_knapsack()

    for i in range(average_iters):
        [solFO, wall_time] = pyswarm_solution(n_particles, iteracions, dimensions, ftol,
                                              options={'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': n_particles, 'p': 1})
        wall_times[0] += wall_time
        solFOs[0] += solFO

        [solFO, wall_time] = ps_bhi_np_solution(n_particles, iteracions, dimensions, ftol,
                                                options={'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': n_particles, 'p': 1})
        wall_times[1] += wall_time
        solFOs[1] += solFO

        [solFO, wall_time] = pso_bhi_jax_solution(n_particles, iteracions, dimensions, ftol,
                                                  options={'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': n_particles, 'p': 1})
        wall_times[2] += wall_time
        solFOs[2] += solFO

    print("\nwall_time pyswarms: {:0.2f}ms".format(1000 * wall_times[0] / average_iters))
    print("wall_time PSO-BHI: {:0.2f}ms (using NumPy)".format(1000 * wall_times[1] / average_iters))
    print("wall_time PSO-BHI: {:0.2f}ms (using JAX.NumPy)".format(1000 * wall_times[2] / average_iters))

    print("\nSolucion Pyswarms: {:0.2f}".format(-solFOs[0] / average_iters))
    print("Solucion Numpy: {:0.2f}".format(-solFOs[1] / average_iters))
    print("Solucion JAX: {:0.2f}".format(-solFOs[2] / average_iters))
    # Change sign because we took negative results from PSO

    # Prints the last run knapsack instance
    print("\nSolución PySwarms:")
    print_solution(solX_PySwarms)
    print("\nSolución NumPy:")
    print_solution(solX_NumPy)
    print("\nSolución JAX.NumPy:")
    print_solution(solX_NumPyJAX)


if __name__ == '__main__':
    compare_solutions()
