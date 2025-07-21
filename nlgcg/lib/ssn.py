# An implementation of the semismooth Newton method following Section 3 of https://mediatum.ub.tum.de/doc/1241413/1241413.pdf

import numpy as np
import logging
from lib.default_values import *

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
        mode: str = "unconstrained",  # "unconstrained" for unconstrained, else for positive solutions
        maximum_iterations: int = 100,
    ) -> None:
        self.K = K
        if all(self.K.shape):
            self.machine_precision = 1e-12
            self.target = target
            self.alpha = alpha
            self.g = lambda u: float(g(u[:-1]))
            self.f = f
            self.grad_f = grad_f
            self.hess_f = hess_f
            self.p = lambda u: -np.array(self.K.T @ self.grad_f(self.K @ u))  # -f'
            self.hessian = lambda u: np.array(
                self.K.T @ self.hess_f(self.K @ u) @ self.K
            )
            self.j = lambda u: float(self.f(self.K @ u) + self.g(u))
            self.M = M
            self.target_norm = np.linalg.norm(self.target, ord=0)
            self.maximum_iterations = maximum_iterations
            if mode == "unconstrained":
                self.Psi = self.Psi_unconstrained
                self.prox = self.prox_unconstrained
                self.grad_prox = self.grad_prox_unconstrained
            else:
                self.Psi = self.Psi_positive
                self.prox = self.prox_positive
                self.grad_prox = self.grad_prox_positive

    def Psi_unconstrained(self, u: np.ndarray) -> np.ndarray:
        # sup_v <p(u),v-u>+g(u)-g(v)
        u = u.copy()
        p = self.p(u)
        constant_part = -np.matmul(p, u) + self.g(u)
        regularization_summand = self.alpha * np.ones(p.shape)
        regularization_summand[-1] = 0  # Last element is not regularized
        norm_multiplier = self.M * np.ones(len(p))
        norm_multiplier[-1] = self.target_norm
        to_maximize = np.multiply(norm_multiplier, np.abs(p) - regularization_summand)
        variable_part = max(0, np.max(to_maximize))
        return constant_part + variable_part

    def Psi_positive(self, u: np.ndarray) -> np.ndarray:
        # sup_v <p(u),v-u>+g(u)-g(v)
        u = u.copy()
        p = self.p(u)
        constant_part = -np.matmul(p, u) + self.g(u)
        regularization_summand = self.alpha * np.ones(p.shape)
        regularization_summand[-1] = 0  # Last element is not regularized
        norm_multiplier = self.M * np.ones(len(p))
        norm_multiplier[-1] = self.target_norm
        to_maximize = np.multiply(norm_multiplier, p - regularization_summand)
        variable_part = max(0, np.max(to_maximize))
        return constant_part + variable_part

    def prox_unconstrained(self, q: np.ndarray) -> np.ndarray:
        q = q.copy()
        to_return = np.zeros(q.shape)
        for i, val in enumerate(q[:-1]):
            if np.abs(val) > self.alpha:
                to_return[i] = val - self.alpha * np.sign(val)
        to_return[-1] = q[-1]  # Last element is not regularized
        return to_return

    def prox_positive(self, q: np.ndarray) -> np.ndarray:
        q = q.copy()
        to_return = np.zeros(q.shape)
        for i, val in enumerate(q[:-1]):
            if val > self.alpha:
                to_return[i] = val - self.alpha
        to_return[-1] = q[-1]  # Last element is not regularized
        return to_return

    def grad_prox_unconstrained(self, q: np.ndarray) -> np.ndarray:
        q = q.copy()
        q[-1] = self.alpha + 1  # Last element is not regularized
        return np.diag(np.where(np.abs(q) > self.alpha, 1, 0))

    def grad_prox_positive(self, q: np.ndarray) -> np.ndarray:
        q = q.copy()
        q[-1] = self.alpha + 1  # Last element is not regularized
        return np.diag(np.where(q > self.alpha, 1, 0))

    def solve(self, tol: float, u_0: np.ndarray) -> np.ndarray:
        # Semismooth Newton method (globalized via line search)
        if not all(self.K.shape):
            logging.debug("Empty input space, retuning u_0")
            return u_0
        theta = tol  # Set initial value for the step length parameter
        Id = np.identity(len(u_0))
        initial_j = self.j(u_0)
        q = u_0  #  + self.p(u_0)
        prox_q = self.prox(q)  # The actual iterate
        k = 0
        while self.Psi(prox_q) > tol or self.j(prox_q) > initial_j:
            right_hand = q - prox_q - self.p(prox_q)
            left_hand = Id + (self.hessian(prox_q) - Id) @ self.grad_prox(q)
            theta = theta / 10
            qdiff = tol + 1
            while qdiff >= tol:
                theta = 2 * theta
                try:
                    direction = np.linalg.solve(left_hand + theta * Id, right_hand)
                except np.linalg.LinAlgError:
                    logging.info(
                        f"SSN in {len(prox_q)} dimensions and tolerance {tol:.3E}: LINEAR SYSTEM NOT SOLVABLE, {self.Psi(prox_q):.3E} achieved"
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
            self.M = float(min(self.M, self.j(prox_q) / self.alpha))
            k += 1
            if k > self.maximum_iterations:
                logging.info(
                    f"SSN in {len(prox_q)} dimensions and tolerance {tol:.3E}: MAX ITERATIONS REACHED, {self.Psi(prox_q):.3E} achieved"
                )
                if self.j(prox_q) <= initial_j:
                    return prox_q
                else:
                    return u_0

        logging.info(
            f"SSN in {len(prox_q)} dimensions converged in {k} iterations to tolerance {tol:.3E}"
        )
        return prox_q


# if __name__ == "__main__":
#     K = np.array([[-1, 2, 0], [3, 0, 0], [-1, -2, -1]])
#     u = np.array([-1, -1, -1])
#     y = np.array([1, 0, 4])
#     sn = SSN_POSITIVE(K, 1, y, 20)
#     print(sn.solve(1e-12, u))
