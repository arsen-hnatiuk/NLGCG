import numpy as np
import os
import jax
import jax.numpy as jnp
import logging
import pickle
import sys
import gc
import matplotlib.pyplot as plt
from pathlib import Path

module_path = Path(__file__).resolve().parent.parent
if module_path not in sys.path:
    sys.path.append(str(module_path))
src_path = (module_path / "src").resolve()
if src_path not in sys.path:
    sys.path.append(str(src_path))

from nlgcg import NLGCG
from src.lib.particle_descent import ParticleDescent
from src.lib.measure import Measure

os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=0 xla_python_client_preallocate=false xla_python_client_mem_fraction=0.5"
)
logging.getLogger().setLevel(logging.INFO)
# init jax
jax.config.update("jax_enable_x64", True)
_ = jnp.zeros(0)

# Generate Data and Define Functions

sigma_space = np.array([0, 1])
omega_space = np.array([[-1, 1], [-1, 1]])
Omega = np.vstack((sigma_space, omega_space))
d = omega_space.shape[0]
variance_exponent = 2.01
alpha = 1e-2
lambda_ = 1000  # Penalty weight for the boundary condition
observation_resolution = 20
test_resolution = 100
max_radius = (Omega[0][1] - Omega[0][0]) / 100
optimum = 26.6427140337648


def define_experiment():
    @jax.jit
    def true_function(x: np.ndarray):
        return (
            jnp.tanh(
                4
                * (
                    0.3
                    - jnp.linalg.norm(
                        x - np.array([0.300000000000000001, 0.3000000000000001])
                    )
                )
            )
            + jnp.tanh(
                12
                * (
                    0.15
                    - jnp.linalg.norm(
                        x + np.array([0.300000000000001, 0.3000000000000001])
                    )
                )
            )
            + 2
        )

    grad_true_function = jax.jit(jax.grad(true_function))
    hess_true_function = jax.jit(jax.hessian(true_function))
    true_function = jax.vmap(true_function)
    grad_true_function = jax.vmap(grad_true_function)
    hess_true_function = jax.vmap(hess_true_function)

    int_function = true_function
    grad_int_function = grad_true_function
    hess_int_function = hess_true_function
    laplacian_int_function = lambda omega: jnp.trace(
        hess_int_function(omega), axis1=1, axis2=2
    )
    bound_function = true_function

    observations_raw = (
        np.array(
            np.meshgrid(
                *(
                    np.linspace(
                        bound[0], bound[1], observation_resolution, endpoint=True
                    )
                    for bound in omega_space
                )
            )
        )
        .reshape(len(omega_space), -1)
        .T
    )
    int_observations = np.array(
        [
            obs
            for obs in observations_raw
            if all(obs != omega_space[0][0]) and all(obs != omega_space[0][1])
        ]
    )
    bound_observations = np.array(
        [
            obs
            for obs in observations_raw
            if any(obs == omega_space[0][0]) or any(obs == omega_space[0][1])
        ]
    )

    test_observations_raw = (
        np.array(
            np.meshgrid(
                *(
                    np.linspace(bound[0], bound[1], test_resolution, endpoint=True)
                    for bound in omega_space
                )
            )
        )
        .reshape(len(omega_space), -1)
        .T
    )
    test_int_observations = np.array(
        [
            obs
            for obs in test_observations_raw
            if all(obs != omega_space[0][0]) and all(obs != omega_space[0][1])
        ]
    )
    test_bound_observations = np.array(
        [
            obs
            for obs in test_observations_raw
            if any(obs == omega_space[0][0]) or any(obs == omega_space[0][1])
        ]
    )

    int_target = int_function(int_observations) ** 3 - laplacian_int_function(
        int_observations
    )
    bound_target = np.sqrt(lambda_) * bound_function(bound_observations)
    target = np.hstack((int_target, bound_target))
    observations = np.vstack((int_observations, bound_observations))

    test_int_target = int_function(test_int_observations) ** 3 - laplacian_int_function(
        test_int_observations
    )
    test_bound_target = np.sqrt(lambda_) * bound_function(test_bound_observations)
    test_target = np.hstack((test_int_target, test_bound_target))
    test_observations = np.vstack((test_int_observations, test_bound_observations))

    constant_dim = len(target)
    kernel_dim = len(target) + len(int_target)

    test_constant_dim = len(test_target)
    test_kernel_dim = len(test_target) + len(test_int_target)

    @jax.jit
    def adjusted_sigmoid(x):
        return 2 * jnp.exp(2 * x) / (1 + jnp.exp(2 * x)) - 1

    @jax.jit
    def raw_kernel(omega: np.ndarray, x: np.ndarray):
        # Both inputs 1D
        sigma = omega[0]
        x_omega = omega[1:]
        variance_factor = adjusted_sigmoid(sigma**variance_exponent)
        inner = -(jnp.sum((x_omega - x) ** 2)) / (2 * sigma**2)
        outer = (jnp.exp(inner) * variance_factor) / (np.sqrt(2 * np.pi) ** d)
        return outer

    hess_raw_kernel = jax.jit(jax.hessian(raw_kernel, argnums=1))
    raw_kernel = jax.vmap(raw_kernel, (None, 0), 0)
    hess_raw_kernel = jax.vmap(hess_raw_kernel, (None, 0), 0)

    @jax.jit
    def singleton_kernel(omega: np.ndarray):
        # function, laplacian
        return jnp.hstack(
            [
                raw_kernel(omega, observations),
                jnp.trace(hess_raw_kernel(omega, int_observations), axis1=1, axis2=2),
            ]
        )

    def test_singleton_kernel(omega: np.ndarray):
        # function, laplacian
        return jnp.hstack(
            [
                raw_kernel(omega, test_observations),
                jnp.trace(
                    hess_raw_kernel(omega, test_int_observations), axis1=1, axis2=2
                ),
            ]
        )

    grad_kernel = jax.jit(jax.jacobian(singleton_kernel))
    hess_kernel = jax.jit(jax.hessian(singleton_kernel))

    kernel = jax.vmap(singleton_kernel)
    test_kernel = jax.vmap(test_singleton_kernel)
    grad_kernel = jax.vmap(grad_kernel)
    hess_kernel = jax.vmap(hess_kernel)

    g = jax.jit(lambda w: alpha * jnp.linalg.norm(w, ord=1))

    @jax.jit
    def r(function_and_laplacian: np.ndarray) -> np.ndarray:
        u = function_and_laplacian[: len(target)]  # shape (len(target),)
        laplacian = function_and_laplacian[len(target) :]  # shape (len(int_target),)
        to_return = jnp.hstack(
            (
                u[: len(int_target)] ** 3 - laplacian,
                np.sqrt(lambda_) * u[len(int_target) :],
            )
        )
        return to_return

    def test_r(function_and_laplacian: np.ndarray) -> np.ndarray:
        u = function_and_laplacian[: len(test_target)]  # shape (len(target),)
        laplacian = function_and_laplacian[
            len(test_target) :
        ]  # shape (len(int_target),)
        to_return = jnp.hstack(
            (
                u[: len(test_int_target)] ** 3 - laplacian,
                np.sqrt(lambda_) * u[len(test_int_target) :],
            )
        )
        return to_return

    f = jax.jit(lambda y: 0.5 * jnp.sum((r(y) - target) ** 2) * 4 / len(target))
    test_f = (
        lambda y: 0.5 * jnp.sum((test_r(y) - test_target) ** 2) * 4 / len(test_target)
    )
    grad_f = jax.jit(jax.grad(f))
    hess_f = jax.jit(jax.hessian(f))

    j = lambda u, c: f(
        u.duality_pairing(kernel, kernel_dim)
        + np.hstack((c * np.ones(constant_dim), np.zeros(kernel_dim - constant_dim)))
    ) + g(u.coefficients)

    @jax.jit
    def p_raw(parameters: np.ndarray, c: float, omega: np.ndarray):
        coefficients = parameters[:, 0]
        support = parameters[:, 1:]
        Ku = jnp.tensordot(kernel(support), coefficients, axes=([0], [0]))
        constant_term = jnp.hstack(
            (c * jnp.ones(constant_dim), jnp.zeros(kernel_dim - constant_dim))
        )
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
        return f(
            kernel(omega).T @ weights
            + constant
            * jnp.hstack((jnp.ones(constant_dim), jnp.zeros(kernel_dim - constant_dim)))
        )

    @jax.jit
    def j_N(raw_input: np.ndarray) -> float:
        input = raw_input[:-1].reshape(-1, d + 2)
        weights = input[:, 0]
        return f_N(raw_input) + g(weights)

    grad_f_N = jax.jit(jax.grad(f_N))
    hess_f_N = jax.jit(jax.hessian(f_N))
    grad_j_N = jax.jit(jax.grad(j_N))

    return (
        true_function,
        target,
        kernel,
        g,
        f,
        f_N,
        grad_f,
        hess_f,
        grad_f_N,
        hess_f_N,
        j,
        j_N,
        p,
        grad_p,
        hess_p,
        grad_j_N,
        constant_dim,
        kernel_dim,
        test_f,
        test_kernel,
        test_kernel_dim,
        test_constant_dim,
    )


def define_nlgcg_experiment():
    (
        true_function,
        target,
        kernel,
        g,
        f,
        f_N,
        grad_f,
        hess_f,
        grad_f_N,
        hess_f_N,
        j,
        j_N,
        p,
        grad_p,
        hess_p,
        grad_j_N,
        constant_dim,
        kernel_dim,
        test_f,
        test_kernel,
        test_kernel_dim,
        test_constant_dim,
    ) = define_experiment()
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
        j=j,
        j_N=j_N,
        p=p,
        grad_p=grad_p,
        hess_p=hess_p,
        grad_j_N=grad_j_N,
        alpha=alpha,
        Omega=Omega,
        global_search_resolution=5,
        dual_variable_goodness=0.3,
        constant_dim=constant_dim,
        kernel_dim=kernel_dim,
        newton_tolerance=1e-1,
        max_radius=max_radius,
    )
    return (
        exp_nlgcg,
        true_function,
        test_f,
        test_kernel,
        test_kernel_dim,
        test_constant_dim,
    )


def define_particle_descent_experiment(m=250, a_parameter=1e-5, b_parameter=5e-6):
    (
        true_function,
        target,
        kernel,
        g,
        f,
        f_N,
        grad_f,
        hess_f,
        grad_f_N,
        hess_f_N,
        j,
        j_N,
        p,
        grad_p,
        hess_p,
        grad_j_N,
        constant_dim,
        kernel_dim,
        test_f,
        test_kernel,
        test_kernel_dim,
        test_constant_dim,
    ) = define_experiment()
    exp_particle = ParticleDescent(
        m=m,
        j=j,
        p=p,
        grad_p=grad_p,
        Omega=Omega,
        a_parameter=a_parameter,
        b_parameter=b_parameter,
        kernel=kernel,
        constant_dim=constant_dim,
        kernel_dim=kernel_dim,
        alpha=alpha,
        target=target,
        g=g,
        f=f,
        grad_f=grad_f,
        hess_f=hess_f,
        residual_tolerance=5e-14,
    )
    return (
        exp_particle,
        true_function,
        test_f,
        test_kernel,
        test_kernel_dim,
        test_constant_dim,
    )


def adapt_time(times, residuals, frame=100, resolution=1):
    to_return = []
    last_pos = 0
    last_res = residuals[0]
    for t in range(int(frame / resolution)):
        minimim_time = t * resolution
        maximum_time = (t + 1) * resolution
        added = False
        for i, (res, tim) in enumerate(zip(residuals[last_pos:], times[last_pos:])):
            if tim < maximum_time and tim >= minimim_time:
                to_return.append(res)
                last_res = res
                last_pos += i + 1
                added = True
                break
        if not added:
            to_return.append(last_res)
        if t * resolution >= times[-1]:
            break
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


def create_plots(Nrun: int = 10):
    resolution = 1  # time resolution for the plots (seconds)
    frame_size = 1000  # Time frame tracked for the residuals (seconds)

    # NLGCG
    nlgcg_residuals = []
    nlgcg_tests = []
    nlgcg_supports = []
    nlgcg_converged = 0
    for i in range(Nrun):
        logging.info(f"Running NLGCG (trial {i+1})")
        (
            exp_nlgcg,
            true_function,
            test_f,
            test_kernel,
            test_kernel_dim,
            test_constant_dim,
        ) = define_nlgcg_experiment()
        (
            us_nlgcg,
            cs_nlgcg,
            times_nlgcg,
            supports_nlgcg,
            inner_loop,
            lgcg_lazy,
            lgcg_total,
            objective_values_nlgcg,
            dropped_tot,
            epsilons,
            all_information,
        ) = exp_nlgcg.solve(
            tol=5e-14, temperature=1, log_results=False, optimum=optimum
        )
        test_values_nlgcg = [
            test_f(
                u.duality_pairing(test_kernel, test_kernel_dim)
                + np.hstack(
                    (
                        c * np.ones(test_constant_dim),
                        np.zeros(test_kernel_dim - test_constant_dim),
                    )
                )
            )
            for u, c, in zip(us_nlgcg, cs_nlgcg)
        ]
        best_test_iteration = np.argmin(test_values_nlgcg)
        u_nlgcg = us_nlgcg[best_test_iteration]
        c_nlgcg = cs_nlgcg[best_test_iteration]
        del exp_nlgcg, true_function, us_nlgcg, cs_nlgcg
        jax.clear_caches()
        local_residuals = adapt_time(
            times_nlgcg,
            [obj - optimum for obj in objective_values_nlgcg],
            frame=frame_size,
            resolution=resolution,
        )
        local_test_values = adapt_time(
            times_nlgcg, test_values_nlgcg, frame=frame_size, resolution=resolution
        )
        if local_residuals[-1] < 1e-5:
            nlgcg_converged += 1
        nlgcg_residuals.append(local_residuals)
        nlgcg_tests.append(local_test_values)
        nlgcg_supports.append(supports_nlgcg)
    logging.info(f"NLGCG converged in {(nlgcg_converged/Nrun)*100}% of cases.")

    nlgcg_residuals_mean = np.mean(bring_to_same_length(nlgcg_residuals), axis=0)
    nlgcg_tests_mean = np.mean(bring_to_same_length(nlgcg_tests), axis=0)
    nlgcg_supports_mean = np.mean(bring_to_same_length(nlgcg_supports), axis=0)
    nlgcg_supports_std = np.std(bring_to_same_length(nlgcg_supports), axis=0)

    # Particle Descent Stochastic
    particle_residuals = []
    particle_tests = []
    particle_supports = []
    particle_converged = 0
    for i in range(Nrun):
        success = False
        while not success:
            logging.info(f"Running Particle descent (trial {i+1})")
            (
                exp_particle,
                true_function,
                test_f,
                test_kernel,
                test_kernel_dim,
                test_constant_dim,
            ) = define_particle_descent_experiment()
            (
                us_particle,
                cs_particle,
                objective_values_particle,
                supports_particle,
                times_particle,
                success,
            ) = exp_particle.solve(
                max_time=frame_size, mode="exponential", log_results=False
            )
            test_values_particle_raw = [
                test_f(
                    u.duality_pairing(test_kernel, test_kernel_dim)
                    + np.hstack(
                        (
                            c * np.ones(test_constant_dim),
                            np.zeros(test_kernel_dim - test_constant_dim),
                        )
                    )
                )
                for u, c, in zip(us_particle, cs_particle)
            ]
            best_test_iteration = np.argmin(test_values_particle_raw)
            u_particle = us_particle[best_test_iteration]
            c_particle = cs_particle[best_test_iteration]
            del exp_particle, us_particle, cs_particle
            jax.clear_caches()
        local_residuals = adapt_time(
            times_particle,
            [obj - optimum for obj in objective_values_particle],
            frame=frame_size,
            resolution=resolution,
        )
        # For particle descent, the test values are recorded every 100 iterations
        test_values_particle = []
        for test_val in test_values_particle_raw:
            test_values_particle += [test_val] * 100
        test_values_particle = test_values_particle[: len(times_particle)]
        local_test_values = adapt_time(
            times_particle,
            test_values_particle,
            frame=frame_size,
            resolution=resolution,
        )
        if local_residuals[-1] < 1e-5:
            particle_converged += 1
        particle_residuals.append(local_residuals)
        particle_tests.append(local_test_values)
        particle_supports.append(supports_particle)
    logging.info(
        f"Particle descent converged in {(particle_converged/Nrun)*100}% of cases."
    )

    particle_residuals_mean = np.mean(bring_to_same_length(particle_residuals), axis=0)
    particle_tests_mean = np.mean(bring_to_same_length(particle_tests), axis=0)
    particle_supports_mean = np.mean(bring_to_same_length(particle_supports), axis=0)
    particle_supports_std = np.std(bring_to_same_length(particle_supports), axis=0)

    logging.getLogger().setLevel(logging.WARNING)  # Supress logging

    # Plot residuals
    fig, ax = plt.subplots(figsize=(5, 4))
    names = ["NLGCG", "Particle Descent"]
    styles = ["-", "--"]
    colors = ["tab:blue", "tab:orange"]
    for array, name, style, color in zip(
        [nlgcg_residuals_mean, particle_residuals_mean],
        names,
        styles,
        colors,
    ):
        ax.semilogy(
            resolution * np.arange(len(array)), array, style, label=name, color=color
        )
    ax.fill(
        np.hstack(
            (
                resolution * np.arange(len(particle_residuals_mean)),
                resolution * np.arange(len(particle_residuals_mean))[::-1],
            )
        ),
        np.hstack(
            (
                np.max(bring_to_same_length(particle_residuals), axis=0),
                np.min(bring_to_same_length(particle_residuals), axis=0)[::-1],
            )
        ),
        "tab:orange",
        alpha=0.3,
    )
    ax.fill(
        np.hstack(
            (
                resolution * np.arange(len(nlgcg_residuals_mean)),
                resolution * np.arange(len(nlgcg_residuals_mean))[::-1],
            )
        ),
        np.hstack(
            (
                np.max(bring_to_same_length(nlgcg_residuals), axis=0),
                np.min(bring_to_same_length(nlgcg_residuals), axis=0)[::-1],
            )
        ),
        "tab:blue",
        alpha=0.3,
    )
    plt.ylabel("Objective residual")
    plt.xlabel("Time (s)")
    plt.ylim(1e-10, 1e3)
    # plt.xlim(0, 100)
    ax.legend()
    plt.savefig(results_dir / "residuals.png", bbox_inches="tight")
    plt.close()

    # Plot test scores
    fig, ax = plt.subplots(figsize=(5, 4))
    names = ["NLGCG", "Particle Descent"]
    styles = ["-", "--"]
    colors = ["tab:blue", "tab:orange"]
    for array, name, style, color in zip(
        [nlgcg_tests_mean, particle_tests_mean],
        names,
        styles,
        colors,
    ):
        ax.semilogy(
            resolution * np.arange(len(array)), array, style, label=name, color=color
        )
    ax.fill(
        np.hstack(
            (
                resolution * np.arange(len(particle_tests_mean)),
                resolution * np.arange(len(particle_tests_mean))[::-1],
            )
        ),
        np.hstack(
            (
                np.max(bring_to_same_length(particle_tests), axis=0),
                np.min(bring_to_same_length(particle_tests), axis=0)[::-1],
            )
        ),
        "tab:orange",
        alpha=0.3,
    )
    ax.fill(
        np.hstack(
            (
                resolution * np.arange(len(nlgcg_tests_mean)),
                resolution * np.arange(len(nlgcg_tests_mean))[::-1],
            )
        ),
        np.hstack(
            (
                np.max(bring_to_same_length(nlgcg_tests), axis=0),
                np.min(bring_to_same_length(nlgcg_tests), axis=0)[::-1],
            )
        ),
        "tab:blue",
        alpha=0.3,
    )
    plt.ylabel("Loss on test set")
    plt.xlabel("Time (s)")
    # plt.ylim(1e-10, 100)
    # plt.xlim(0, 100)
    ax.legend()
    plt.savefig(results_dir / "test_values.png", bbox_inches="tight")
    plt.close()

    # Plot supports
    fig, ax = plt.subplots(figsize=(5, 4))
    names = ["NLGCG", "Particle Descent"]
    styles = ["-", "--"]
    for array, name, style, color in zip(
        [
            nlgcg_supports_mean,
            particle_supports_mean,
        ],
        names,
        styles,
        colors,
    ):
        ax.semilogx(np.arange(len(array)), array, style, label=name, color=color)
    ax.fill(
        np.hstack(
            (
                np.arange(len(particle_supports_mean)),
                np.arange(len(particle_supports_mean))[::-1],
            )
        ),
        np.hstack(
            (
                np.array(particle_supports_mean) - np.array(particle_supports_std),
                np.array(particle_supports_mean)[::-1]
                + np.array(particle_supports_std)[::-1],
            )
        ),
        "tab:orange",
        alpha=0.3,
    )
    ax.fill(
        np.hstack(
            (
                np.arange(len(nlgcg_supports_mean)),
                np.arange(len(nlgcg_supports_mean))[::-1],
            )
        ),
        np.hstack(
            (
                np.array(nlgcg_supports_mean) - np.array(nlgcg_supports_std),
                np.array(nlgcg_supports_mean)[::-1]
                + np.array(nlgcg_supports_std)[::-1],
            )
        ),
        "tab:blue",
        alpha=0.3,
    )
    plt.ylabel("Support points")
    plt.xlabel("Iterations")
    # plt.ylim(1e-12, 100)
    # plt.xlim(0, 100)
    ax.legend()
    plt.savefig(results_dir / "supports.png", bbox_inches="tight")
    plt.close()

    # for u, c, name in zip(
    #     [u_nlgcg, u_particle],
    #     [c_nlgcg, c_particle],
    #     ["solution_nlgcg", "solution_particle"],
    # ):
    #     with open(f"{results_dir}/{name}.pkl", "wb") as file:
    #         pickle.dump((u, c), file)

    # with open(results_dir / "solution_nlgcg.pkl", "rb") as file:
    #     u_nlgcg, c_nlgcg = pickle.load(file)
    # with open(results_dir / "solution_particle.pkl", "rb") as file:
    #     u_particle, c_particle = pickle.load(file)

    # Plot pruning of Particle Descent
    thresholds = np.linspace(0, 0.0001, 26)
    max_coef = np.max(np.abs(u_particle.coefficients))
    pruned_test_vals = []
    pruned_support_sizes = []
    for threshold in thresholds:
        pruned_indices = np.abs(u_particle.coefficients) > threshold * max_coef
        pruned_support_sizes.append(np.sum(pruned_indices))
        pruned_u = Measure(
            support=u_particle.support[pruned_indices],
            coefficients=u_particle.coefficients[pruned_indices],
        )
        pruned_test_vals.append(
            test_f(
                pruned_u.duality_pairing(test_kernel, test_kernel_dim)
                + np.hstack(
                    (
                        c_particle * np.ones(test_constant_dim),
                        np.zeros(test_kernel_dim - test_constant_dim),
                    )
                )
            )
        )
    fig, ax1 = plt.subplots(figsize=(5, 4))
    ax2 = ax1.twinx()
    (p1,) = ax1.semilogy(
        thresholds, pruned_test_vals, "-", label="Loss on test set", color="darkred"
    )
    (p2,) = ax2.plot(
        thresholds, pruned_support_sizes, "-", label="Support size", color="lightpink"
    )
    ax1.set_ylabel("Loss on test set")
    ax2.set_ylabel("Support size")
    ax1.set_xlabel("Pruning threshold")
    ax1.legend(handles=[p1, p2], loc="upper center")
    plt.savefig(results_dir / "particle_pruning.png", bbox_inches="tight")
    plt.close()

    # Plot the function with reconstruction error
    resolution = 100
    a_1 = np.linspace(omega_space[0][0], omega_space[0][1], resolution, endpoint=False)
    a_2 = np.linspace(omega_space[1][0], omega_space[1][1], resolution, endpoint=False)
    x, y = np.meshgrid(a_1, a_2)
    points = np.array(list(zip(x.flatten(), y.flatten())))
    vals = true_function(points).reshape((resolution, resolution))

    def plot_kernel(omega: np.ndarray):
        sigma = omega[0]
        x_omega = omega[1:]
        inner = -(jnp.linalg.norm(x_omega - points, axis=1) ** 2) / (2 * sigma**2)
        outer = (jnp.exp(inner) * sigma**variance_exponent) / (np.sqrt(2 * np.pi) ** d)
        return outer

    pred_vals_nlgcg = u_nlgcg.duality_pairing(jax.vmap(plot_kernel)).reshape(
        (resolution, resolution)
    ) + c_nlgcg * np.ones((resolution, resolution))
    pred_vals_particle = u_particle.duality_pairing(jax.vmap(plot_kernel)).reshape(
        (resolution, resolution)
    ) + c_particle * np.ones((resolution, resolution))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    contour1 = ax1.contourf(x, y, vals, levels=100)
    fig.colorbar(contour1, ax=ax1)
    ax1.set_xlim(omega_space[0][0], omega_space[0][1])
    ax1.set_ylim(omega_space[1][0], omega_space[1][1])
    ax1.set_xlabel("True function")
    for pred_vals, name, u, ax in zip(
        [pred_vals_nlgcg, pred_vals_particle],
        ["NLGCG", "Particle Descent"],
        [u_nlgcg, u_particle],
        [ax2, ax3],
    ):
        contour = ax.contourf(x, y, pred_vals, levels=100)
        fig.colorbar(contour, ax=ax)
        for i, point in enumerate(u.support):
            if u.coefficients[i] < 0:
                color = "b"
            else:
                color = "r"
            ax.add_patch(
                plt.Circle(
                    (point[1], point[2]),
                    radius=point[0],
                    color=color,
                    fill=False,
                    alpha=0.5,
                )
            )
        ax.set_xlim(omega_space[0][0], omega_space[0][1])
        ax.set_ylim(omega_space[1][0], omega_space[1][1])
        ax.set_xlabel(f"Predicted function {name}")
    plt.savefig(results_dir / f"predictions.png", bbox_inches="tight")
    plt.close()

    logging.getLogger().setLevel(logging.INFO)  # Reinstate logging

    # Best test value
    test_value_nlgcg = test_f(
        u_nlgcg.duality_pairing(test_kernel, test_kernel_dim)
        + np.hstack(
            (
                c_nlgcg * np.ones(test_constant_dim),
                np.zeros(test_kernel_dim - test_constant_dim),
            )
        )
    )
    test_value_particle = test_f(
        u_particle.duality_pairing(test_kernel, test_kernel_dim)
        + np.hstack(
            (
                c_particle * np.ones(test_constant_dim),
                np.zeros(test_kernel_dim - test_constant_dim),
            )
        )
    )
    logging.info(
        f"Best test value NLGCG: {test_value_nlgcg:.3E}, Particle Descent: {test_value_particle:.3E}"
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
        default="results/pinn",
        help="Directory for plots",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    create_plots(Nrun=args.Nrun)
