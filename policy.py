import numpy as np
import torch


class EpsilonGreedyPolicy:
    '''
    TODO: refactor this
    '''
    def __init__(self, num_states, num_actions, epsilon, Q=None):
        if Q is None:
            self._Q = np.zeros((num_states, num_actions))
        else:
            self._Q = Q
        self._eps = epsilon

    @property
    def Q(self):
        # support read-only
        return np.copy(self._Q)

    def query_Q_probs(self, s=None, a=None):
        Q_probs = np.zeros(self._Q.shape)
        for s in range(self._Q.shape[0]):
            Q_probs[s, :] = self.query_Q_probs(s)
        if s is None and a is None:
            return Q_probs
        elif a is None:
            return Q_probs[s, :]
        else:
            return Q_probs[s, a]


    def _query_Q_probs(self, s, a=None):
        num_actions = self._Q.shape[1]
        probs = np.ones(num_actions, dtype=float) * self._eps / num_actions
        ties = np.flatnonzero(self._Q[s, :] == self._Q[s, :].max())
        if a is None:
            best_a = np.random.choice(ties)
            probs[best_a] += 1. - self._eps
            return probs
        else:
            if a in ties:
                probs[a] += 1. - self._eps
            return probs[a]

    def choose_action(self, s):
        probs = self._query_Q_probs(s)
        return np.random.choice(len(probs), p=probs)

    def update_Q_val(self, s, a, val):
        self._Q[s,a] = val


class GreedyPolicy:
    def __init__(self, num_states, num_actions, Q=None):
        if Q is None:
            # start with random policy
            self._Q = np.zeros((num_states, num_actions))
        else:
            # in case we want to import e-greedy
            self._Q = Q

    @property
    def Q(self):
        # support read-only
        return np.copy(self._Q)

    def query_Q_probs(self, s=None, a=None):
        Q_probs = np.zeros(self._Q.shape).astype(np.float)
        for s in range(self._Q.shape[0]):
            ties = np.flatnonzero(self._Q[s, :] == self._Q[s, :].max())
            a = np.random.choice(ties)
            Q_probs[s, a] = 1.0
        if s is None and a is None:
            return Q_probs
        elif a is None:
            return Q_probs[s, :]
        else:
            return Q_probs[s, a]

    def choose_action(self, s):
        ties = np.flatnonzero(self._Q[s, :] == self._Q[s, :].max())
        return np.random.choice(ties)

    def get_opt_actions(self):
        opt_actions = np.zeros(self._Q.shape[0])
        for s in range(opt_actions.shape[0]):
            opt_actions[s] = self.choose_action(s)
        return opt_actions

    def update_Q_val(self, s, a, val):
        self._Q[s,a] = val


class StochasticPolicy:
    def __init__(self, num_states, num_actions, Q=None):
        if Q is None:
            # start with random policy
            self._Q = np.zeros((num_states, num_actions))
        else:
            # in case we want to import e-greedy
            # make Q non negative to be useful as probs
            self._Q = Q

    @property
    def Q(self):
        # support read-only
        return np.copy(self._Q)

    def query_Q_probs(self, s=None, a=None, laplacian_smoothing=True):
        '''
        returns:
            probability distribution of actions over all states
        '''
        if laplacian_smoothing:
            LAPLACIAN_SMOOTHER = 0.01
            L = (np.max(self._Q, axis=1) - np.min(self._Q, axis=1))* LAPLACIAN_SMOOTHER
            Q = self._Q - np.expand_dims(np.min(self._Q, axis=1) - L, axis=1)
        else:
            Q = self._Q - np.expand_dims(np.min(self._Q, axis=1), axis=1)
        Q_sum = np.sum(Q, axis=1)
        # if zero, we give uniform probs with some gaussian noise
        num_actions = self._Q.shape[1]
        Q[Q_sum==0, :] = 1.
        Q_sum[Q_sum==0] = num_actions
        Q_probs = Q / np.expand_dims(Q_sum, axis=1)
        Q_probs[Q_sum==0, :] += np.random.normal(0, 1e-4, num_actions)

        if s is None and a is None:
            return Q_probs
        elif a is None:
            return Q_probs[s, :]
        else:
            return Q_probs[s, a]


    def choose_action(self, s, laplacian_smoothing=True):
        probs = self.query_Q_probs(s, laplacian_smoothing=laplacian_smoothing)
        return np.random.choice(len(probs), p=probs)

    def update_Q_val(self, s, a, val):
        self._Q[s,a] = val


class RandomPolicy:
    def __init__(self, num_states, num_actions):
        self._Q_probs = np.ones((num_states, num_actions), dtype=float) / num_actions

    @property
    def Q(self):
        # support read-only
        return np.copy(self._Q_probs)

    def choose_action(self, s):
        probs = self._Q_probs[s, :]
        return np.random.choice(len(probs), p=probs)


class RandomPolicy2:
    def __init__(self, choices):
        self._choices = choices

    def choose_action(self, s):
        """ sample uniformly """
        return np.random.choice(self._choices)


import torch
from torch import nn, optim
from torch.autograd import Variable


class LinearQ(nn.Module):
    """Docstring for LinearQ. """

    def __init__(self, phi, k):
        """TODO: to be defined1.

        Parameters
        ----------
        phi : basis function for (s, a)
        k : feature dimension
        """
        super().__init__()
        self._phi = phi
        self._l1 = nn.Linear(k, 1)


    def forward(self, s, a):
        """
        predict Q(s,a)
        """
        x = self._phi(s, a)
        out = self._l1(x)
        return out


    def choose_action(self, s):
        """
        argmax_a Q(s, a)
        """
        Q_hat = np.array([self.forward(self._phi(s, a)) for a in range(n_actions)])
        ties = np.flatnonzero(Q_hat = Q_hat.max())
        return np.random.choice(ties)



class LinearQ2(object):
    """Docstring for LinearQ. """

    def __init__(self, phi, W):
        """TODO: to be defined1.

        Parameters
        ----------
        phi : basis function of (s, a)
        W : TODO, optional
        W : TODO, optional


        """
        self._phi = phi
        self._W = W


    def predict(self, s, a=None):
        """TODO: Docstring for predict.

        only works for discrete action space

        Parameters
        ----------
        s : state

        Returns
        -------
        Q(s, a)

        """
        if a is None:
            return self._W.T.dot(self._phi(s))
        else:
            return self._W.T.dot(self._phi(s, a))


    def choose_action(self, s):
        Q_hat = self.predict(s)
        ties = np.flatnonzero(Q_hat = Q_hat.max())
        return np.random.choice(ties)



