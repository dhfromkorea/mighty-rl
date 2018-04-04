# inspired by: https://github.com/stober/lspi/blob/master/src/lspi.py
# heavily modified

import numpy as np
from numpy.linalg import inv, norm
import logging


from policy import LinearQ2


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
        self._reward_fn = reward_fn
        self._D = None
        self._W_hat = None


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
        check dimensionality of everything

        """

        self._D = D
        A_hat = np.zeros((self._p, self._p))
        b_hat = np.zeros((1, self._p))
        # perhaps can be done in one step?
        s = np.vstack(self._D[:, 0])
        a = np.vstack(self._D[:, 1])

        if self._reward_fn is not None:
            # postprocess reward for IRL
            r = np.vstack([self._reward_fn(s,a) for s, a in zip(s, a)])
            logging.debug("modified reward: {}".format(r))
        else:
            r = np.vstack(self._D[:, 2])
            logging.debug("original reward: {}".format(r))

        s_next = np.vstack(self._D[:, 3])

        phi = self._phi(s, a)

        a_next = np.vstack(np.apply_along_axis(pi.choose_action, 1, s_next))
        phi_next = self._phi(s_next, a_next)

        phi_delta = phi - self._gamma * phi_next
        # A_hat: p x p matrix
        A_hat = phi.T.dot(phi_delta)
        # b_hat: 1 x p matrix
        b_hat = r.T.dot(phi)
        A_hat += self._eps * np.identity(self._p)
        # xi_hat: p x p (1 x p)^T = p x 1
        W_hat = inv(A_hat).dot(b_hat.T)

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
        p : int
            dimension of phi
        q : int
            dimension of psi
        pi : Policy
            policy to evaluate

        Returns
        -------
        xi_hat = xi_hat

        TODO
        - vectorize this
        - phi(s, a) or phi(s) when to use
        - what phi or psi to use?
        - check dimensionality of everytthing


        """
        self._D = D
        A_hat = np.zeros((self._q, self._q))
        b_hat = np.zeros((self._q, self._p))

        # perhaps can be done in one step?
        s = np.vstack(self._D[:, 0])
        a = np.vstack(self._D[:, 1])
        r = np.vstack(self._D[:, 2])
        s_next = np.vstack(self._D[:, 3])

        psi = self._psi(s, a)

        a_next = np.vstack(np.apply_along_axis(pi.choose_action, 1, s_next))
        psi_next = self._psi(s_next, a_next)

        psi_delta = psi - self._gamma * psi_next

        # A_hat: q x q matrix
        A_hat = psi.T.dot(psi_delta)
        # b_hat: q x p matrix
        b_hat = psi.T.dot(self._phi(s, a))
        A_hat += self._eps * np.identity(self._q)
        # xi_hat: q x p matrix
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

        # return p x 1 vector
        # xi_hat q x p
        # psi 1 x q

        return self._xi_hat.T.dot(self._psi(s0, a0).T)


class LSPI(object):
    """Docstring for LSPI. """
    def __init__(self, D, action_list, p, phi, gamma, precision, eps, W_0, reward_fn):
        """TODO: to be defined1.

        Parameters
        ----------
        D : TODO
        action)list : list of valid action indices
        collet_D : fn that collects extra samples
        p : dimension of phi
        phi : TODO
        gamma : TODO
        precision : convergence threshold
        eps : make A invertible
        W_0 : initial weight
        reward_fn : (optional) non-default reward fn to simulate MDP\R
        """
        self._D = D
        self._action_list = action_list
        #self._collect_D = collect_D
        self._p = p
        self._phi = phi
        self._gamma = gamma
        self._precision = precision
        self._eps = eps
        self._W_0 = W_0
        self._W = None
        self._reward_fn = reward_fn


    def solve(self):
        W = self._W_0
        D = self._D


        t = 0

        while True:
            t+=1
            print("lspi iter count ", t)
            W_old = W
            # update D
            lstd_q = LSTDQ(p=self._p,
                           phi=self._phi,
                           gamma=self._gamma,
                           eps=self._eps,
                           reward_fn=self._reward_fn)
            pi = LinearQ2(action_list=self._action_list,
                          W=W_old,
                          phi=self._phi)
            W = lstd_q.fit(D=self._D, pi=pi)
            #D = self._collect_D(D, W)
            logging.info("lspi W {}".format(W))
            logging.info("lspi W old {}".format(W_old))
            logging.info("lspi norm {}".format(norm(W - W_old, 2)))
            if norm(W - W_old, 2) < self._precision:
                break
        # save
        self._W = W
        return W





