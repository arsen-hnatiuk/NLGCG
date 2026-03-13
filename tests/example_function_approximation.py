#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os

os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=0"
)
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# init jax
_ = jnp.zeros(0)

import pickle
import logging
import time
import sys
import matplotlib.pyplot as plt
from pathlib import Path

module_path = Path(__file__).resolve().parent.parent
if module_path not in sys.path:
    sys.path.append(str(module_path))
src_path = (module_path / "src").resolve()
if src_path not in sys.path:
    sys.path.append(str(src_path))

from nlgcg import NLGCG
from src.lib.measure import Measure
from src.lib.ssn import SSN
from src.lib.particle_descent import ParticleDescent
from src.lib.adaptive_refinement import AdaptiveRefinement

# Function Approximation

logging.getLogger().setLevel(logging.INFO)  # Supress logging

# Generate Data and Define Functions

sigma_space = np.array([0, 2])
omega_space = np.array([[-1, 1], [-1, 1]])
Omega = np.vstack((sigma_space, omega_space))
d = omega_space.shape[0]
variance_exponent = 0.1
alpha = 1e-3
observation_resolution = 20
endpoint = False
max_radius = 1e-2
max_inner_loop = 25
cg_lambda = 1e-3
cg_iterations = 100

true_function = lambda x: np.minimum(1 - np.abs(x[:, 0]), 1 - np.abs(x[:, 1]))

max_radius = (Omega[0][1] - Omega[0][0]) / 100

observations = (
    np.array(
        np.meshgrid(
            *(
                np.linspace(
                    bound[0] + 1e-5,
                    bound[1] - np.sqrt(3) * 1e-5,
                    observation_resolution,
                    endpoint=endpoint,
                )
                for bound in omega_space
            )
        )
    )
    .reshape(len(omega_space), -1)
    .T
)


@jax.jit
def singleton_kernel(omega: np.ndarray):
    sigma = omega[0]
    x = omega[1:]
    inner = -jnp.sum((x - observations) ** 2, axis=1) / (2 * sigma**2)
    outer = (jnp.exp(inner) * sigma ** (variance_exponent)) / (
        jnp.sqrt(2 * jnp.pi) ** d
    )
    return outer


grad_kernel = jax.jit(jax.jacobian(singleton_kernel))
hess_kernel = jax.jit(jax.hessian(singleton_kernel))
_ = grad_kernel(np.ones(len(Omega)))
_ = hess_kernel(np.ones(len(Omega)))

kernel = jax.vmap(singleton_kernel)
grad_kernel = jax.vmap(grad_kernel)
hess_kernel = jax.vmap(hess_kernel)

target = true_function(observations)


@jax.jit
def g(w: np.ndarray) -> float:
    return alpha * jnp.linalg.norm(w, ord=1)


grad_g = jax.jit(jax.grad(g))
_ = grad_g(jnp.ones(1))


@jax.jit
def f(y: np.ndarray) -> float:
    return 0.5 * jnp.sum((y - target) ** 2)


grad_f = jax.jit(jax.grad(f))
hess_f = jax.jit(jax.hessian(f))
_ = grad_f(jnp.zeros(target.shape[0]))
_ = hess_f(jnp.zeros(target.shape[0]))

j = lambda u, c: f(u.duality_pairing(kernel) + c * np.ones(target.shape)) + g(
    u.coefficients
)


@jax.jit
def p_raw(parameters: np.ndarray, c: float, omega: np.ndarray):
    coefficients = parameters[:, 0]
    support = parameters[:, 1:]
    Ku = jnp.tensordot(kernel(support), coefficients, axes=([0], [0]))
    constant_term = c * jnp.ones(len(target))
    return singleton_kernel(omega) @ -grad_f(Ku + constant_term)


grad_p_raw = jax.jit(jax.grad(p_raw, argnums=2))
hess_p_raw = jax.jit(jax.hessian(p_raw, argnums=2))

p_raw = jax.jit(jax.vmap(p_raw, in_axes=(None, None, 0), out_axes=0))
grad_p_raw = jax.jit(jax.vmap(grad_p_raw, in_axes=(None, None, 0), out_axes=0))
hess_p_raw = jax.jit(jax.vmap(hess_p_raw, in_axes=(None, None, 0), out_axes=0))

p = lambda u, c: lambda omega: p_raw(u.to_matrix(len(Omega)), c, omega)
grad_p = lambda u, c: lambda omega: grad_p_raw(u.to_matrix(len(Omega)), c, omega)
hess_p = lambda u, c: lambda omega: hess_p_raw(u.to_matrix(len(Omega)), c, omega)


# Parameterized versions of f and j
@jax.jit
def f_N(raw_input: np.ndarray) -> float:
    constant = raw_input[-1]
    input = raw_input[:-1].reshape(-1, d + 2)
    weights = input[:, 0]
    omega = input[:, 1:]
    return f(kernel(omega).T @ weights + constant * jnp.ones(target.shape))


@jax.jit
def j_N(raw_input: np.ndarray) -> float:
    input = raw_input[:-1].reshape(-1, d + 2)
    weights = input[:, 0]
    return f_N(raw_input) + g(weights)


grad_f_N = jax.jit(jax.grad(f_N))
hess_f_N = jax.jit(jax.hessian(f_N))
grad_j_N = jax.jit(jax.grad(j_N))
hess_j_N = jax.jit(jax.hessian(j_N))

# Define functions where the derivatives are taken wrt regularized (weights) and non-regularized(support + constant) parameters


@jax.jit
def f_N_(weights: np.ndarray, support_constant: np.ndarray) -> float:
    constant = support_constant[-1]
    omega = support_constant[:-1].reshape(-1, d + 1)
    return f(kernel(omega).T @ weights + constant * jnp.ones(target.shape))


@jax.jit
def j_N_(weights: np.ndarray, support_constant: np.ndarray) -> float:
    return f_N_(weights, support_constant) + g(weights)


grad_f_N_reg = jax.jit(jax.grad(f_N_, argnums=0))
hess_f_N_reg = jax.jit(jax.hessian(f_N_, argnums=0))
grad_j_N_reg = jax.jit(jax.grad(j_N_, argnums=0))
hess_j_N_reg = jax.jit(jax.hessian(j_N_, argnums=0))

grad_f_N_non_reg = jax.jit(jax.grad(f_N_, argnums=1))
hess_f_N_non_reg = jax.jit(jax.hessian(f_N_, argnums=1))
grad_j_N_non_reg = jax.jit(jax.grad(j_N_, argnums=1))
hess_j_N_non_reg = jax.jit(jax.hessian(j_N_, argnums=1))

optimum = 0.033949312592952266


def create_particle_matrix():

    # NLGCG
    exp = NLGCG(
        target=target,
        kernel=kernel,
        g=g,
        f=f,
        f_N=f_N,
        grad_f=grad_f,
        hess_f=hess_f,
        grad_f_N=grad_f_N,
        hess_f_N=hess_f_N,
        grad_f_N_non_reg=grad_f_N_non_reg,
        j=j,
        j_N=j_N,
        j_N_=j_N_,
        p=p,
        grad_p=grad_p,
        hess_p=hess_p,
        grad_j_N=grad_j_N,
        hess_j_N=hess_j_N,
        alpha=alpha,
        Omega=Omega,
        global_search_resolution=5,
        dual_variable_goodness=0.3,
        constant_dim=len(target),
        kernel_dim=len(target),
        newton_tolerance=2e-2,
    )

    (
        u,
        c,
        times,
        supports,
        inner_loop,
        lgcg_lazy,
        lgcg_total,
        objective_values,
        dropped_tot,
        epsilons,
    ) = exp.solve(tol=5e-14, max_radius=max_radius, temperature=0.1)

    print(
        f"found optimimum with value {objective_values[-1]} (difference to ref {optimum} is {objective_values[-1] - optimum})"
    )
    print(f"constant {c}")
    print(f"solution {u}")

    # logging.getLogger().setLevel(logging.WARN)

    # runs_per_Nparticle = {12: 50, 24: 50, 50: 20, 100: 10, 200: 5}
    runs_per_Nparticle = {2000: 1}
    success_per_Nparticle = {key: 0 for key in runs_per_Nparticle.keys()}

    for Nparticle, Nruns in runs_per_Nparticle.items():
        # Particle Gradient Descent (a_parameter does not matter so much because of linesearch)
        print(f"running trials with {Nparticle} particles")
        a_parameter = 0.0000001
        b_parameter_factor = 0.1
        b_parameter = b_parameter_factor * a_parameter
        exp = ParticleDescent(
            m=Nparticle,
            j=j,
            p=p,
            grad_p=grad_p,
            Omega=Omega,
            a_parameter=a_parameter,
            b_parameter=b_parameter,
            kernel=kernel,
            constant_dim=len(target),
            kernel_dim=len(target),
            alpha=alpha,
            target=target,
            g=g,
            f=f,
            grad_f=grad_f,
            hess_f=hess_f,
            do_linesearch=True,
            residual_tolerance=1e-16,
        )

        # run Nruns to determine success probability
        for it in range(Nruns):
            u, c, objective_values, supports, times, success = exp.solve(
                max_iters=int(1e6), max_time=20 * 60, mode="uniform"
            )

            print(
                f"\t({it+1} of {Nruns}): found optimum with value {objective_values[-1]} (residual {objective_values[-1] - optimum})"
            )
            # print(u.to_matrix()[np.abs(u.coefficients) > 1e-3, :])

            success = objective_values[-1] - optimum < 1e-3
            success_per_Nparticle[Nparticle] += int(success)

            if success:
                plt.semilogy(
                    np.arange(len(objective_values)),
                    np.array(objective_values) - optimum,
                )
                plt.show()
                plt.close()

        print(
            {
                Npart: success_per_Nparticle[Npart] / Nrun
                for Npart, Nrun in runs_per_Nparticle.items()
            }
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--Nrun", type=int, default=10, help="Number runs of stochastic methods"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/example_signal",
        help="Directory for plots",
    )
    parser.add_argument(
        "--particle-matrix",
        action="store_true",
        help="Generate a particle matrix plot instead of algorithm comparison",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.particle_matrix:
        create_particle_matrix()
    else:
        create_plots(Nrun=args.Nrun, results_dir=results_dir)
