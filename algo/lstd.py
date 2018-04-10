# inspired by: https://github.com/stober/lspi/blob/master/src/lspi.py
# heavily modified

import itertools
import sys

import numpy as np
from numpy.linalg import inv, norm, cond, solve, matrix_rank, lstsq
import logging
from algo.policy import LinearQ2
import scipy.sparse.linalg as spla


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


    def fit(self, s, a, r, s_next, a_next, pi):
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

        A_hat = np.zeros((self._p, self._p))
        A_hat += self._eps * np.identity(self._p)
        b_hat = np.zeros((self._p, 1))

        phi = self._phi(s, a)
        phi_next = self._phi(s_next, a_next)
        phi_delta = phi - self._gamma * phi_next
        # A_hat: p x p matrix
        A_hat = phi.T.dot(phi_delta)
        # b_hat: p x 1 matrix
        b_hat = phi.T.dot(r)

        #logging.debug("A_hat\n{}".format(A_hat))
        #logging.debug("condition number of A_hat\n{}".format(cond(A_hat)))


        #W_hat = inv(A_hat).dot(b_hat.T)
        if matrix_rank(A_hat) == self._p:
            # use LU decomposition
            # requires to be full rank
            W_hat = solve(A_hat, b_hat)
        else:
            import pdb;pdb.set_trace()
            W_hat = lstsq(A_hat, b_hat)[0]

        self._W_hat = W_hat
        return W_hat


    def fit2(self, D, pi):
        """TODO: Docstring for learn.
        iterative

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
        # to help with invertibility of A
        A_hat += self._eps * np.identity(self._p)
        b_hat = np.zeros((1, self._p))

        for (s, a, r, s_next, _) in self._D:
            phi = self._phi(s, a)
            print("phi", phi)

            # policy to evaluate
            a_next = pi.choose_action(s_next)
            phi_delta = phi - self._gamma * self._phi(s_next, a_next)
            A_hat += phi.dot(phi_delta.T)

            # just use reward?
            if self._reward_fn is not None:
                logging.debug("original reward: {}".format(r))
                r = self._reward_fn(s, a)
                logging.debug("modified reward: {}".format(r))
            b_hat += phi.dot(r)

        print("cond a_hat {}".format(cond(A_hat)))
        if matrix_rank(A_hat) == p:
            # use LU decomposition
            # requires to be full rank
            W_hat = solve(A_hat, b_hat.T)
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
        #xi_hat = inv(A_hat).dot(b_hat)
        xi_hat = solve(A_hat, b_hat)
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
    def __init__(self, D, action_list, p, phi, gamma, precision, eps, W_0, reward_fn, max_iter=10):
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
        max_iter : int
            The maximum number of iterations force termination.
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
        self._max_iter = max_iter


    def solve(self):
        W = self._W_0
        D = self._D
        W_list = []

        # preprocessing
        logging.info("fitting D of the dimension:\n{}".format(D.shape))
        s = np.vstack(self._D[:, 0])
        s_next = np.vstack(self._D[:, 3])
        a = np.vstack(self._D[:, 1])

        if self._reward_fn is not None:
            r = np.vstack([self._reward_fn(s,a) for s, a in zip(s, a)])
            logging.debug("modified reward: {}".format(r))
        else:
            r = np.vstack(self._D[:, 2])
            logging.debug("original reward: {}".format(r))

        for t in itertools.count():
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

            a_next = np.vstack(np.apply_along_axis(pi.choose_action, 1, s_next))
            W = lstd_q.fit(s, a, r, s_next, a_next, pi=pi)
            W_list.append(W)
            logging.debug("lspi W {}".format(W))
            logging.debug("lspi W old {}".format(W_old))
            logging.info("lspi norm {}".format(norm(W - W_old, 2)))
            if t > self._max_iter or norm(W - W_old) < self._precision:
                break
        # save
        self._W = W

        return W, W_list





