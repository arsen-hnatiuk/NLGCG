import numpy as np
import logging
from lib.default_values import *
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
_ = jnp.zeros(0)

logging.basicConfig(
    level=logging.DEBUG,
)


class SSN:
    def __init__(
        self,
        K: np.ndarray,
        alpha: float,
        target: np.ndarray,
        M: float,
        g: Callable,
        f: Callable,
        grad_f: Callable,
        hess_f: Callable,
        invariable_kernel: np.ndarray,
        mode: str = "unconstrained",  # "unconstrained" for unconstrained, else for positive solutions
        maximum_iterations: int = 1000,
        regularization: str = "mixed",
    ) -> None:
        self.K = K
        if all(self.K.shape):
            self.target = target
            self.alpha = alpha
            self.f = f
            self.grad_f = grad_f
            self.hess_f = hess_f
            self.invariable_kernel = invariable_kernel
            self.p = lambda u: -np.array(
                self.K.T @ self.grad_f(self.invariable_kernel + self.K @ u)
            )  # -f'
            self.hessian = lambda u: np.array(
                self.K.T @ self.hess_f(self.invariable_kernel + self.K @ u) @ self.K
            )
            self.j = lambda u: float(
                self.f(self.invariable_kernel + self.K @ u) + self.g(u)
            )
            self.M = M
            self.target_norm = np.mean(np.abs(self.target))
            self.maximum_iterations = maximum_iterations
            self.regularization = regularization
            self.machine_precision = 5e-14
            if regularization == "mixed":
                self.g = lambda u: float(g(u[:-1]))
            else:
                self.g = g
            if mode == "unconstrained":
                self.Psi = self.Psi_unconstrained
                self.prox = self.prox_unconstrained
                self.grad_prox = self.grad_prox_unconstrained
            else:
                self.Psi = self.Psi_positive
                self.prox = self.prox_positive
                self.grad_prox = self.grad_prox_positive

    def Psi_unconstrained(self, u: np.ndarray) -> np.ndarray:
        # sup_v <p(u),v-u>+g(u)-g(v), adjusted for numerical stability
        u = u.copy()
        u_norm = np.linalg.norm(u, ord=1)
        p = self.p(u)
        if u_norm:
            constant_part = (-np.matmul(p, u) + self.g(u)) / u_norm
        else:
            constant_part = 0
        regularization_summand = self.alpha * np.ones(p.shape)
        if self.regularization == "mixed":
            regularization_summand[-1] = 0  # Last element is not regularized
        norm_multiplier = np.ones(len(p))
        to_maximize = np.multiply(norm_multiplier, np.abs(p) - regularization_summand)
        variable_part = max(0, np.max(to_maximize))
        return constant_part + variable_part

    def Psi_positive(self, u: np.ndarray) -> np.ndarray:
        # sup_v <p(u),v-u>+g(u)-g(v), adjusted for numerical stability
        u = u.copy()
        u_norm = np.linalg.norm(u, ord=1)
        p = self.p(u)
        if u_norm:
            constant_part = (-np.matmul(p, u) + self.g(u)) / u_norm
        else:
            constant_part = 0
        regularization_summand = self.alpha * np.ones(p.shape)
        if self.regularization == "mixed":
            regularization_summand[-1] = 0  # Last element is not regularized
        norm_multiplier = np.ones(len(p))
        to_maximize = np.multiply(norm_multiplier, p - regularization_summand)
        variable_part = max(0, np.max(to_maximize))
        return constant_part + variable_part

    def prox_unconstrained(self, q: np.ndarray) -> np.ndarray:
        q = q.copy()
        to_return = np.zeros(q.shape)
        for i, val in enumerate(q):
            if np.abs(val) > self.alpha:
                to_return[i] = val - self.alpha * np.sign(val)
        if self.regularization == "mixed":
            to_return[-1] = q[-1]  # Last element is not regularized
        return to_return

    def prox_positive(self, q: np.ndarray) -> np.ndarray:
        q = q.copy()
        to_return = np.zeros(q.shape)
        for i, val in enumerate(q):
            if val > self.alpha:
                to_return[i] = val - self.alpha
        if self.regularization == "mixed":
            to_return[-1] = q[-1]  # Last element is not regularized
        return to_return

    def grad_prox_unconstrained(self, q: np.ndarray) -> np.ndarray:
        q = q.copy()
        if self.regularization == "mixed":
            q[-1] = self.alpha + 1  # Last element is not regularized
        return np.diag(np.where(np.abs(q) > self.alpha, 1, 0))

    def grad_prox_positive(self, q: np.ndarray) -> np.ndarray:
        q = q.copy()
        if self.regularization == "mixed":
            q[-1] = self.alpha + 1  # Last element is not regularized
        return np.diag(np.where(q > self.alpha, 1, 0))

    def solve(self, tol: float, u_0: np.ndarray, log_results: bool) -> np.ndarray:
        # Semismooth Newton method (globalized via line search)
        if not all(self.K.shape):
            logging.debug("Empty input space, retuning u_0")
            return u_0
        theta = tol  # Set initial value for the step length parameter
        Id = np.identity(len(u_0))
        initial_j = self.j(u_0)
        q = u_0  #  + self.p(u_0)
        prox_q = self.prox(q)  # The actual iterate
        psi_val = min(self.Psi(prox_q), self.Psi(q))
        k = 0
        while psi_val > tol:
            if k > self.maximum_iterations:
                if log_results:
                    logging.warning(
                        f"SSN in {len(prox_q)} dimensions and tolerance {tol:.3E}: MAX ITERATIONS REACHED, {psi_val:.3E} achieved"
                    )
                if self.j(prox_q) <= initial_j:
                    return prox_q
                else:
                    return u_0
            right_hand = q - prox_q - self.p(prox_q)
            left_hand = Id + (self.hessian(prox_q) - Id) @ self.grad_prox(q)
            theta = theta / 10
            qdiff = tol + 1
            while qdiff >= self.machine_precision:
                theta = 2 * theta
                try:
                    direction = np.linalg.solve(left_hand + theta * Id, right_hand)
                except np.linalg.LinAlgError:
                    if log_results:
                        logging.warning(
                            f"SSN in {len(prox_q)} dimensions and tolerance {tol:.3E}: LINEAR SYSTEM NOT SOLVABLE, {psi_val:.3E} achieved"
                        )
                    if self.j(prox_q) <= initial_j:
                        return prox_q
                    else:
                        return u_0
                qnew = q - direction
                prox_qnew = self.prox(qnew)
                qdiff = self.j(prox_qnew) - self.j(prox_q)
            q = qnew
            prox_q = prox_qnew
            psi_val = self.Psi(prox_q)
            k += 1

        if log_results:
            logging.info(
                f"SSN in {len(prox_q)} dimensions converged in {k} iterations to tolerance {tol:.3E}"
            )
        if k == 0:
            if self.j(q) < self.j(prox_q):
                return q
        return prox_q


# if __name__ == "__main__":
#     K = np.array([[-1, 2, 0], [3, 0, 0], [-1, -2, -1]])
#     u = np.array([-1, -1, -1])
#     y = np.array([1, 0, 4])
#     sn = SSN_POSITIVE(K, 1, y, 20)
#     print(sn.solve(1e-12, u))
