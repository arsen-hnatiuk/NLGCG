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
        minimum_iterations: int = 0,
        mode: str = "unconstrained",
        # mode: "unconstrained" for unconstrained, else for positive solutions
    ) -> None:
        self.K = K
        if all(self.K.shape):
            self.machine_precision = 1e-12
            self.target = target
            self.alpha = alpha
            self.g = get_default_g(self.alpha)
            self.f = get_default_f(self.K, self.target)
            self.p = get_default_p(self.K, self.target)  # -f'
            self.hessian = get_default_hessian(self.K)
            self.j = lambda u: self.f(u) + self.g(u)
            self.M = M
            self.minimum_iterations = minimum_iterations
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
        p = self.p(u)
        constant_part = -np.matmul(p, u) + self.g(u)
        variable_part = max(0, self.M * (np.max(np.absolute(p)) - self.alpha))
        return constant_part + variable_part

    def Psi_positive(self, u: np.ndarray) -> np.ndarray:
        # sup_v <p(u),v-u>+g(u)-g(v)
        p = self.p(u)
        constant_part = -np.matmul(p, u) + self.g(u)
        variable_part = max(0, self.M * (np.max(p) - self.alpha))
        return constant_part + variable_part

    def prox_unconstrained(self, q: np.ndarray, alpha: float) -> np.ndarray:
        to_return = np.zeros(q.shape)
        for i, val in enumerate(q):
            if np.abs(val) > alpha:
                to_return[i] = val - alpha * np.sign(val)
        return to_return

    def prox_positive(self, q: np.ndarray, alpha: float) -> np.ndarray:
        to_return = np.zeros(q.shape)
        for i, val in enumerate(q):
            if val > alpha:
                to_return[i] = val - alpha
        return to_return

    def grad_prox_unconstrained(self, q: np.ndarray, alpha: float) -> np.ndarray:
        return np.diag(np.where(np.abs(q) > alpha, 1, 0))

    def grad_prox_positive(self, q: np.ndarray, alpha: float) -> np.ndarray:
        return np.diag(np.where(q > alpha, 1, 0))

    def solve(self, tol: float, u_0: np.ndarray) -> np.ndarray:
        # Semismooth Newton method (globalized via line search)
        if not all(self.K.shape):
            logging.debug("Empty input space, retuning u_0")
            return u_0
        theta = tol  # Set initial value for the step length parameter
        Id = np.identity(len(u_0))
        initial_j = self.j(u_0)
        q = u_0 + self.p(u_0)
        prox_q = self.prox(q, self.alpha)  # The actual iterate
        k = 0
        while k - 1 < self.minimum_iterations:
            while self.Psi(prox_q) > tol or self.j(prox_q) > initial_j:
                right_hand = q - prox_q - self.p(prox_q)
                left_hand = Id + (self.hessian - Id) @ self.grad_prox(q, self.alpha)
                theta = theta / 10
                direction = np.linalg.solve(left_hand + theta * Id, right_hand)
                qnew = q - direction
                prox_qnew = self.prox(qnew, self.alpha)

                # Backtracking line search
                qdiff = self.j(prox_qnew) - self.j(prox_q)
                while qdiff >= tol:
                    theta = 2 * theta
                    try:
                        direction = np.linalg.solve(left_hand + theta * Id, right_hand)
                    except np.linalg.LinAlgError:
                        logging.debug(
                            f"SSN in {len(prox_q)} dimensions and tolerance {tol:.3E}: LINEAR SYSTEM NOT SOLVABLE, {self.Psi(prox_q):.3E} achieved"
                        )
                        return prox_q
                    qnew = q - direction
                    prox_qnew = self.prox(qnew, self.alpha)
                    qdiff = self.j(prox_qnew) - self.j(prox_q)

                q = qnew
                prox_q = prox_qnew
                k += 1
                if k > 1000:
                    logging.debug(
                        f"SSN in {len(prox_q)} dimensions and tolerance {tol:.3E}: MAX ITERATIONS REACHED, {self.Psi(prox_q):.3E} achieved"
                    )
                    return prox_q
            last_tol = tol
            tol = max(tol / 2, self.machine_precision)
            k += 1
        tol = last_tol

        logging.debug(
            f"SSN in {len(prox_q)} dimensions converged in {k} iterations to tolerance {tol:.3E}"
        )
        return prox_q


# if __name__ == "__main__":
#     K = np.array([[-1, 2, 0], [3, 0, 0], [-1, -2, -1]])
#     u = np.array([-1, -1, -1])
#     y = np.array([1, 0, 4])
#     sn = SSN_POSITIVE(K, 1, y, 20)
#     print(sn.solve(1e-12, u))
