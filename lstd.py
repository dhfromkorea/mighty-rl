import numpy as np
from numpy.linalg import inv

# inspired: https://github.com/stober/lspi/blob/master/src/lspi.py

class LSTDQ(object):
    """Docstring for LSTD. """

    def __init__(self, p, phi, gamma, eps, reward_fn=None):
        """TODO: to be defined1.

        Parameters
        ----------
        D : trajectory data
        p : dimension of phi
        phi : basis function for reward
        gamma : (0, 1)
        eps : small positive value to make A invertible
        reward_fn : (optional) non-default reward fn to simulate MDP\R
        """

        self._p = p
        self._phi = phi
        self._gamma = gamma
        self._eps = eps
        self._W_hat = None
        self._reward_fn = reward_fn


    def fit(self, D, pi):
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

        self._D = D
        A_hat = np.zeros((self._p, self._p))
        # make A almost always invertible
        # unless eps is A's eigenvalue
        A_hat += self._eps * np.identity(self._p)
        b_hat = np.zeros((self._p, 1))

        for traj in self._D:
            for (s, a, r, s_next, done) in traj:
                phi = self._phi(s, a)

                # policy to evaluate
                a_next = pi.choose_action(s_next)
                phi_delta = phi - self._gamma * self._phi(s_next, a_next)
                A_hat += phi.dot(phi_delta.T)

                # just use reward?

                if self._reward_fn is not None:
                    logging.debug("modified reward: {}".format(r))
                    r = self._reward_fn(s, a)
                    logging.debug("modified reward: {}".format(r))
                b_hat += phi.dot(r)

        W_hat = inv(A_hat).dot(b_hat)
        self._W_hat = W_hat
        return W_hat


    def estimate_Q(self, s0, a0):
        """estimate Q^pi(s,a)

        essentially policy evaluation

        Parameters
        ----------

        Returns
        -------
        Q_hat : Q estimate given a fixed policy pi
        """
        return self._W_hat.T.dot(self.phi(s0, a0))


class LSTDMu(LSTDQ):
    """Docstring for LSTDMu. """

    def __init__(self, p, phi, q, psi, gamma, eps):
        """TODO: to be defined1.

        Parameters
        ----------
        D : trajectory data
        p : dimension of phi
        phi : basis function for reward
        psi : basis function for feature expectation
        gamma : (0, 1)
        eps : small positive value to make A invertible
        W : W to evaluate
        """
        super().__init__(p=p, phi=phi, gamma=gamma, eps=eps)
        self._psi = psi
        self._q = q
        self._xi_hat = None

    def fit(self, D, pi):
        """estimate xi to compute mu

        assuminng action-value function mu(s, a)
        is linearly parametrized by xi
        such that mu(s, a) = Q_phi(s, a) = xi^T psi(s)

        Parameters
        ----------
        p : dimension of phi
        q : dimension of psi
        pi : policy to evaluate

        Returns
        -------
        xi_hat = xi_hat

        TODO
        - vectorize this
        - phi(s, a) or phi(s) when to use
        - what phi or psi to use?


        """
        self._D = D
        A_hat = np.zeros((self._q, self._q))
        b_hat = np.zeros((self._q, self._p))

        # perhaps can be done in one step?

        for traj in self._D:
            for (s, a, r, s_next, done) in traj:
                psi = self._psi(s, a)

                # policy to evaluate
                a_next = pi.choose_action(s_next)
                psi_delta = psi - self._gamma * self._psi(s_next, a_next)

                A_hat += psi.dot(psi_delta.T)
                # just use reward?
                b_hat += psi.dot(self._phi(s, a).T)

        # make A almost always invertible
        # unless eps is A's eigenvalue
        A_hat += self._eps * np.identity(self._q)

        xi_hat = inv(A_hat).dot(b_hat)
        self._xi_hat = xi_hat
        return xi_hat

    def predict(self, s0, a0):
        """estimate mu

        Parameters
        ----------

        Returns
        -------
        mu_hat = mu_hat

        TODO
        - what if no action?
        """

        return self._xi_hat.T.dot(self._psi(s0, a0))


class LSPI(object):
    """Docstring for LSPI. """

    def __init__(self, D, p, phi, gamma, eps, W_0, reward_fn):
        """TODO: to be defined1.

        Parameters
        ----------
        D : TODO
        collet_D : fn that collects extra samples
        p : dimension of phi
        phi : TODO
        gamma : TODO
        eps : TODO
        W_0 : initial weight
        reward_fn : (optional) non-default reward fn to simulate MDP\R
        """
        self._D = D
        #self._collect_D = collect_D
        self._p = p
        self._phi = phi
        self._gamma = gamma
        self._eps = eps
        self._W_0 = W_0
        self._W = None
        self._reward_fn = reward_fn


    def solve(self):
        W = self._W_0
        D = self._D

        while norm(W - W_old, 2) > self._eps:
            W_old = W
            # update D
            W = LSTDQ(D=D,
                     p=self._p,
                     phi=self._phi,
                     gamma=self._gamma,
                     eps=self._eps,
                     W=W,
                     reward_fn=reward_fn)
            #D = self._collect_D(D, W)
        # save
        self._W = W
        return W





