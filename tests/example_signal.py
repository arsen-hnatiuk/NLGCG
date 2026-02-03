#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy as sp
from typing import Callable, Union
import os
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=0"
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

# init jax
_ = jnp.zeros(0)

import pickle
import logging
import time
import sys
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, ".")  # TODO: not necessary if there is a local install

from nlgcg.nlgcg import NLGCG
from nlgcg.lib.measure import Measure
from nlgcg.lib.ssn import SSN
from nlgcg.lib.particle_descent import ParticleDescent
from nlgcg.lib.adaptive_refinement import AdaptiveRefinement


# Signal Processing

logging.getLogger().setLevel(logging.INFO) # Supress logging

# Generate Data and Define Functions

observation_resolution = 120
Omega = np.array([[0, observation_resolution//2]])
alpha = 1e-1
true_sources = np.array([[3.125], [7.], [np.sqrt(179.)]])
true_weights = np.array([-1., 0.7, 0.5])
true_measure = Measure(support=true_sources, coefficients=true_weights)


max_radius = (Omega[0][1] - Omega[0][0]) / 100.

observations = np.arange(0, 1, 1/observation_resolution)

@jax.jit
def singleton_kernel(omega: float):
    return jnp.sin(2*np.pi*omega*observations)

grad_kernel = jax.jit(jax.jacobian(singleton_kernel))
hess_kernel = jax.jit(jax.hessian(singleton_kernel))
_ = grad_kernel(np.ones(len(Omega)))
_ = hess_kernel(np.ones(len(Omega)))

kernel = jax.vmap(singleton_kernel)
grad_kernel = jax.vmap(grad_kernel)
hess_kernel = jax.vmap(hess_kernel)

target = true_measure.duality_pairing(kernel)


@jax.jit
def g(w: np.ndarray) -> float:
    return alpha * jnp.linalg.norm(w, ord=1)
grad_g = jax.jit(jax.grad(g))
_ = grad_g(jnp.ones(1))

@jax.jit
def f(y: np.ndarray) -> float:
    return 0.5 * jnp.sum((y - target)**2)
grad_f = jax.jit(jax.grad(f))
hess_f = jax.jit(jax.hessian(f))
_ = grad_f(jnp.zeros(target.shape[0]))
_ = hess_f(jnp.zeros(target.shape[0]))

j = lambda u, c: f(u.duality_pairing(kernel)+c*np.ones(target.shape)) + g(u.coefficients)

@jax.jit
def p_raw(parameters: np.ndarray, c: float, omega: np.ndarray):
    coefficients = parameters[:,0]
    support = parameters[:,1:]
    Ku = jnp.tensordot(kernel(support), coefficients, axes=([0], [0]))
    constant_term = c*jnp.ones(len(target))
    return singleton_kernel(omega) @ -grad_f(Ku+constant_term)

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
    input = raw_input[:-1].reshape(-1, Omega.shape[0]+1)
    weights = input[:,0]
    omega = input[:,1:]
    return f(kernel(omega).T@weights+constant*jnp.ones(target.shape))

@jax.jit
def j_N(raw_input: np.ndarray) -> float:
    input = raw_input[:-1].reshape(-1, Omega.shape[0]+1)
    weights = input[:,0]
    return f_N(raw_input) + g(weights)

grad_f_N = jax.jit(jax.grad(f_N))
hess_f_N = jax.jit(jax.hessian(f_N))
grad_j_N = jax.jit(jax.grad(j_N))
hess_j_N = jax.jit(jax.hessian(j_N))


# Define functions where the derivatives are taken wrt regularized (weights) and non-regularized(support + constant) parameters

@jax.jit
def f_N_(weights: np.ndarray, support_constant: np.ndarray) -> float:
    constant = support_constant[-1]
    omega = support_constant[:-1].reshape(-1, Omega.shape[0])
    return f(kernel(omega).T@weights + constant*jnp.ones(target.shape))

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



optimum = 0.21975385192787872

# Experiments

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
        newton_tolerance=2e-2
    )

    u, c, times, supports, inner_loop, lgcg_lazy, lgcg_total, objective_values, dropped_tot, epsilons = exp.solve(
        tol=5e-14, max_radius=max_radius, temperature=0.1)

    print(f"found optimimum with value {objective_values[-1]} (difference to ref {optimum} is {objective_values[-1] - optimum})")
    print(f"times: {np.array(times)}")
    print(f"dropped: {dropped_tot}")
    print(f"solution {u}")

    logging.getLogger().setLevel(logging.INFO)

    # Particle Gradient Descent
    a_parameter = 0.001
    exp = ParticleDescent(
        m=200,
        j=j,
        p=p,
        grad_p=grad_p,
        Omega=Omega,
        a_parameter=a_parameter,
        b_parameter=a_parameter/2,
        kernel=kernel,
        constant_dim=len(target),
        kernel_dim=len(target),
        alpha=alpha,
        target=target,
        g=g,
        f=f,
        grad_f=grad_f,
        hess_f=hess_f,
        ssn_steps=100,
        do_linesearch=True
    )
    u, c, objective_values, supports, times, success = exp.solve(max_iters=int(1e6), max_time=5*60, mode="uniform")

    print(f"found optimimum with value {objective_values[-1]} (difference to ref {optimum} is {objective_values[-1] - optimum})")

    print(u.to_matrix()[np.abs(u.coefficients) > 1e-3, :])


def create_plots(Nrun: int = 10, results_dir: Path = Path("results/signal_example")):
    exp_nlgcg = NLGCG(
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
        global_search_resolution=10,
        dual_variable_goodness=0.3,
        constant_dim=len(target),
        kernel_dim=len(target),
        newton_tolerance=2e-2
    )

    exp_particle = ParticleDescent(
        m=50,
        j=j,
        p=p,
        grad_p=grad_p,
        Omega=Omega,
        a_parameter=0.001,
        b_parameter=0.001/2,
        kernel=kernel,
        constant_dim=len(target),
        kernel_dim=len(target),
        alpha=alpha,
        target=target,
        g=g,
        f=f,
        grad_f=grad_f,
        hess_f=hess_f,
        ssn_steps=100,
        do_linesearch=True
    )

    exp_adaptive = AdaptiveRefinement(
        observations=observations,
        j=j,
        p=p,
        grad_p=grad_p,
        hess_p=hess_p,
        Omega=Omega,
        kernel=kernel,
        constant_dim=len(target),
        kernel_dim=len(target),
        alpha=alpha,
        target=target,
        g=g,
        f=f,
        grad_f=grad_f,
        hess_f=hess_f,
        ssn_steps=1000,
    )

    def adapt_time(times, residuals, frame=100, resolution=1):
        to_return = []
        last_pos = 0
        last_res = residuals[0]
        for t in range(frame):
            for i, (res, tim) in enumerate(zip(residuals[last_pos:], times[last_pos:])):
                if tim > t+resolution:
                    to_return.append(last_res)
                    if tim  - t - resolution < resolution:
                        last_pos += i
                    break
                else:
                    last_res = res
                    to_return.append(last_res)
        return to_return

    def bring_to_same_length(arrays):
        max_length = max(len(arr) for arr in arrays)
        new_arrays = []
        for arr in arrays:
            if len(arr) < max_length:
                last_val = arr[-1]
                arr = arr + [last_val] * (max_length - len(arr))
            new_arrays.append(arr)
        return new_arrays


    logging.getLogger().setLevel(logging.CRITICAL) # Supress logging


    # deterministic NLCG
    print(f"Running deterministic NLCG")
    u, c, times_det_nlgcg, supports_det_nlgcg, inner_loop, lgcg_lazy, lgcg_total, objective_values_det_nlgcg, dropped_tot, epsilons = exp_nlgcg.solve(
        tol=5e-14, max_radius=max_radius, temperature=0.1, mode="deterministic")
    residuals_det_nlgcg = adapt_time(times_det_nlgcg, [obj - optimum for obj in objective_values_det_nlgcg], frame=1000, resolution=1)

    print(f"NLCG converged up to residual {objective_values_det_nlgcg[-1] - optimum}")

    # Adaptive refinement
    print(f"Running adaptive refinement")
    cells_dict, vertices_dict, vertices, u, objective_values_adaptive, times_adaptive, actives, supports_adaptive = exp_adaptive.solve(max_iters=200)
    residuals_adaptive = adapt_time(times_adaptive, [obj - optimum for obj in objective_values_adaptive], frame=1000, resolution=1)

    print(f"Adaptive refinement converged up to residual {objective_values_adaptive[-1] - optimum}")

    # NLGCG stochastic
    nlgcg_residuals = []
    nlgcg_supports = []
    for i in range(Nrun):
        print(f"Running stochastic NLCG (trial {i})")
        u, c, times_nlgcg, supports_nlgcg, inner_loop, lgcg_lazy, lgcg_total, objective_values_nlgcg, dropped_tot, epsilons = exp_nlgcg.solve(
            tol=5e-14, max_radius=max_radius, temperature=0.1)
        local_residuals = adapt_time(times_nlgcg, [obj - optimum for obj in objective_values_nlgcg], frame=1000, resolution=1)
        nlgcg_residuals.append(local_residuals)
        nlgcg_supports.append(supports_nlgcg)
        print(f"Stochastic NLCG converged up to residual {objective_values_nlgcg[-1] - optimum}")


    nlgcg_residuals_mean = np.mean(bring_to_same_length(nlgcg_residuals), axis=0)
    nlgcg_residuals_std = np.std(bring_to_same_length(nlgcg_residuals), axis=0)
    nlgcg_supports_mean = np.mean(bring_to_same_length(nlgcg_supports), axis=0)
    nlgcg_supports_std = np.std(bring_to_same_length(nlgcg_supports), axis=0)


    # Particle Descent Stochastic
    particle_residuals = []
    particle_supports = []
    for i in range(Nrun):
        print(f"Running Particle descent (trial {i})")
        max_iter = int(1e4) #int(1e6)
        u, c, objective_values_particle, supports_particle, times_particle, success = exp_particle.solve(max_iters=max_iter, mode="uniform")
        local_residuals = adapt_time(times_particle, [obj - optimum for obj in objective_values_particle], frame=1000, resolution=1)
        particle_residuals.append(local_residuals)
        particle_supports.append(supports_particle)
        print(f"Particle descent converged up to residual {objective_values_particle[-1] - optimum}")

    particle_residuals_filtered = []
    converged_frac = 0
    for i in range(len(particle_residuals)):
        if particle_residuals[i][-1] < 1e-8:
            particle_residuals_filtered.append(particle_residuals[i])
            converged_frac += 1/Nrun

    if converged_frac > 1.9 / Nrun:
        particle_residuals_mean = np.mean(bring_to_same_length(particle_residuals_filtered), axis=0)
        particle_residuals_std = np.std(bring_to_same_length(particle_residuals_filtered), axis=0)
    else:
        particle_residuals_mean = np.mean(bring_to_same_length(particle_residuals), axis=0)
        particle_residuals_std = np.std(bring_to_same_length(particle_residuals), axis=0)

    particle_supports_mean = np.mean(bring_to_same_length(particle_supports), axis=0)
    particle_supports_std = np.std(bring_to_same_length(particle_supports), axis=0)
    print(f"Particle descent converged in {converged_frac*100}% of cases.")


    # Plot residuals
    fig, ax = plt.subplots(figsize=(12,10))
    names = ["NLGCG", "Adaptive Refinement", "SNLGCG", "Particle Descent"]
    styles = ["-", "-.", "--", ":"]
    for array, name, style in zip([residuals_det_nlgcg, residuals_adaptive, nlgcg_residuals_mean, particle_residuals_mean], names, styles):
        ax.semilogy(np.arange(len(array)), array, style, label=name);

    ax.fill(np.hstack((np.arange(len(particle_residuals_mean)),
                       np.arange(len(particle_residuals_mean))[::-1])),
            np.hstack((np.array(particle_residuals_mean) - np.array(particle_residuals_std),
                       np.array(particle_residuals_mean)[::-1] + np.array(particle_residuals_std)[::-1])),
            'red', alpha=0.3);
    ax.fill(np.hstack((np.arange(len(nlgcg_residuals_mean)),
                       np.arange(len(nlgcg_residuals_mean))[::-1])),
            np.hstack((np.array(nlgcg_residuals_mean) - np.array(nlgcg_residuals_std),
                       np.array(nlgcg_residuals_mean)[::-1] + np.array(nlgcg_residuals_std)[::-1])),
            'green', alpha=0.3);

    plt.ylabel("Objective residual")
    plt.xlabel("Time (s)")
    plt.ylim(1e-12, 100)
    # plt.xlim(0, 100)
    ax.legend()
    plt.savefig(results_dir / "residuals.pdf")
    plt.close()

    # Plot supports
    fig, ax = plt.subplots(figsize=(12,10))
    names = ["NLGCG", "Adaptive Refinement", "SNLGCG", "Particle Descent"]
    styles = ["-", "-.", "--", ":"]
    for array, name, style in zip([supports_det_nlgcg, supports_adaptive, nlgcg_supports_mean, particle_supports_mean], names, styles):
        ax.semilogx(np.arange(len(array)), array, style, label=name);

    ax.fill(np.hstack((np.arange(len(particle_supports_mean)),
                       np.arange(len(particle_supports_mean))[::-1])),
            np.hstack((np.array(particle_supports_mean)-np.array(particle_supports_std),
                       np.array(particle_supports_mean)[::-1]+np.array(particle_supports_std)[::-1])),
            'red', alpha=0.3);
    ax.fill(np.hstack((np.arange(len(nlgcg_supports_mean)),
                       np.arange(len(nlgcg_supports_mean))[::-1])),
            np.hstack((np.array(nlgcg_supports_mean)-np.array(nlgcg_supports_std),
                       np.array(nlgcg_supports_mean)[::-1]+np.array(nlgcg_supports_std)[::-1])),
            'green', alpha=0.3);
    plt.ylabel("Support points")
    plt.xlabel("Iterations")
    # plt.ylim(1e-12, 100)
    # plt.xlim(0, 100)
    ax.legend()
    plt.savefig(results_dir / "support_size.pdf")
    plt.close()


    # Plot number of coefficients to otimize
    fig, ax = plt.subplots(figsize=(12,10))
    names = ["NLGCG", "Adaptive Refinement", "SNLGCG"]
    styles = ["-", "-.", "--"]
    for array, name, style in zip([supports_det_nlgcg, actives, nlgcg_supports_mean], names, styles):
        ax.semilogx(np.arange(len(array)), array, style, label=name);

    ax.fill(np.hstack((np.arange(len(nlgcg_supports_mean)),
                       np.arange(len(nlgcg_supports_mean))[::-1])),
            np.hstack((np.array(nlgcg_supports_mean)-np.array(nlgcg_supports_std),
                       np.array(nlgcg_supports_mean)[::-1]+np.array(nlgcg_supports_std)[::-1])),
            'green', alpha=0.3);
    plt.ylabel("Number of coefficients to optimize");
    plt.xlabel("Iterations");
    # plt.ylim(1e-12, 100);
    # plt.xlim(0, 100);
    ax.legend();
    plt.savefig(results_dir / "support_size_2.pdf")
    plt.close()


    # Plots
    a = np.arange(Omega[0][0], Omega[0][1], 0.001)
    p_u = p(u, c)
    vals = p_u(a)
    plt.plot(a,vals);
    plt.axvline(x=-1, linestyle="-", c="r", label="Support");
    plt.axvline(x=-1, linestyle="-", c="g", label="Truth");
    for i, pos in enumerate(u.support):
        ymax = 0.5+(u.coefficients[i]*alpha*0.25+0.05*np.sign(u.coefficients[i]))
        plt.axvline(x=pos, ymin=0.5,ymax=ymax, linestyle="-", c="r");
    for point, weight in  zip(true_sources, true_weights):
        ymax = 0.5+(weight*alpha*0.25+0.05*np.sign(weight))
        plt.axvline(x=point, ymin=0.5,ymax=ymax,alpha=0.5, linestyle="-", c="g");

    plt.xlabel("Frequency");
    plt.ylim(-alpha*1.1,alpha*1.1);
    plt.xlim(0, 50);
    plt.legend();
    plt.savefig(results_dir / "sources.pdf")
    plt.close()


    # Plot the measured signal
    def signal(x, sources, weights, c):
        # Input is 2D array of shape (number of points, Omega dimension)
        weighted_signal = c * np.ones(x.shape)
        for point, weight in zip(sources, weights):
            weighted_signal += weight*np.sin(2*np.pi*point*x).flatten()
        return weighted_signal

    a = np.arange(0,1,0.001)
    true_signal = signal(a, true_sources, true_weights, 0.)
    plt.plot(a, true_signal);
    plt.xlabel("Time");
    plt.savefig(results_dir / "signal.pdf")

    a = np.arange(0, 1, 0.001)
    predicted_signal = signal(a, u.support, u.coefficients, c)
    error = true_signal - predicted_signal
    print(f"L2 error: {np.linalg.norm(error)/np.sqrt(len(a))}, Linf error: {np.max(np.abs(error))}")
    plt.plot(a, np.abs(true_signal-predicted_signal));
    plt.xlabel("Time");
    plt.savefig(results_dir / "signal_error.pdf")
    plt.close()


    # Plot residuals
    residuals = np.array(objective_values_det_nlgcg) - optimum
    plt.figure(figsize=(12,5))
    plt.semilogy(np.array(range(len(residuals)-1)), residuals[:-1], linestyle="-.", color="green");
    # for interval in intervals:
    #     plt.fill_between(interval, 0, 60, hatch="/", color="gray", alpha=0.2);
    plt.ylim(1e-17, 60);
    # plt.xlim(2500, 3000);
    plt.ylabel("Objective residual");
    plt.xlabel("Total iterations");
    plt.savefig(results_dir / "residulas.pdf")
    plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--Nrun", type=int, default=10, help='Number runs of stochastic methods')
    parser.add_argument("--results-dir", type=str, default="results/example_signal", help='Directory for plots')
    parser.add_argument("--particle-matrix", action="store_true", help='Generate a particle matrix plot instead of algorithm comparison')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.particle_matrix:
        create_particle_matrix()
    else:
        create_plots(Nrun=args.Nrun, results_dir=results_dir)
