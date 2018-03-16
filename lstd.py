import numpy as np
from numpy.linalg import inv


class LSTD(object):
    """Docstring for LSTD. """

    def __init__(self, D, k, phi, gamma, eps, pi):
        """TODO: to be defined1.

        Parameters
        ----------
        D : trajectory data
        k : dimension of features for each state
        phi : basis function for reward
        gamma : (0, 1)
        eps : small positive value to make A invertible
        pi : policy to evaluate


        """
        self._D = D
        self._k = k
        self._phi = phi
        self._psi = psi
        self._gamma = gamma
        self._eps = eps
        self._pi = pi


    def esimate_W(self):
        """TODO: Docstring for learn.

        assuminng action-value function Q(s,a)
        is linearly parametrized by W
        such that Q = W^T phi(s)

        Parameters
        ----------

        Returns
        -------
        TODO
        this is LSTD_Q

        """
        A_hat = np.zeros(self._k, self._k)
        # make A almost always invertible
        # unless eps is A's eigenvalue
        A_hat += self._eps * np.identity(self._k)
        b_hat = np.zeros(self._k, 1)

        for (s, a, s_next, r) in self._D:
            phi = self._phi(s, a)
            phi_delta = self._phi(s, a) - self._gamma * self._phi(s_next, a)
            A_hat += phi.dot(phi_delta.T)

            # just use reward?
            b_hat += self._phi(s, a) * r

        W_hat = inv(A_hat).dot(b_hat)
        return W_hat


    def estimate_Q(self, s, a):
        """estimate Q^pi(s,a)

        essentially policy evaluation

        Parameters
        ----------

        Returns
        -------
        Q_hat : Q estimate given a fixed policy pi
        """

        W_hat = self.esimate_W()
        return W_hat.T.dot(self.phi(s, a))



class LSTD_Mu(LSTD):
    """Docstring for LSTD_Mu. """

    def __init__(self, D, k, phi, psi, gamma, eps, pi):
        """TODO: to be defined1.

        Parameters
        ----------
        D : trajectory data
        k : dimension of features for each state
        phi : basis function for reward
        psi : basis function for feature expectation
        gamma : (0, 1)
        eps : small positive value to make A invertible
        pi : policy to evaluate
        """
        super().__init__(self, D, k, phi, psi, gamma, eps, pi)
        self._psi = psi


    def estimate_xi(self):
        """estimate xi to compute mu

        assuminng action-value function mu(s, a)
        is linearly parametrized by xi
        such that mu(s, a) = Q_phi(s, a) = xi^T psi(s)

        Parameters
        ----------

        Returns
        -------
        xi_hat = xi_hat

        TODO
        - vectorize this
        - phi(s, a) or phi(s) when to use
        - what phi or psi to use?


        """
        A_hat = np.zeros(self._k, self._k)
        # make A almost always invertible
        # unless eps is A's eigenvalue
        A_hat += self._eps * np.identity(self._k)
        b_hat = np.zeros(self._k, 1)

        # perhaps can be done in one step?
        for (s, a, s_next, r) in self._D:
            psi = self._psi(s, a)
            psi_delta = psi - self._gamma * self._psi(s_next, a)
            A_hat += psi.dot(psi_delta.T)

            # just use reward?
            b_hat += psi.dot(self._phi(s, a))

        xi_hat = inv(A_hat).dot(b_hat)
        return xi_hat


    def estimate_mu(self, s):
        """estimate mu

        Parameters
        ----------

        Returns
        -------
        mu_hat = mu_hat
        """

        xi_hat = self.estimate_mu()
        return xi_hat.T.dot(self._psi(s))



