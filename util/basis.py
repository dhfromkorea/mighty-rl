import numpy as np
import sklearn.pipeline
from sklearn.kernel_approximation import RBFSampler
import sklearn.preprocessing


class RBFKernel(object):
    """Docstring for RBFKernel. """

    def __init__(self, env, n_component=25, include_action_to_basis=False, include_action=False):
        """TODO: to be defined1.

        Parameters
        ----------
        env : TODO
        n_component : TODO, optional
        """
        states = np.array([env.observation_space.sample() for x in range(10000)])
        actions = np.array([env.action_space.sample() for x in range(10000)]).reshape(10000, 1)
        # giving state action
        if include_action_to_basis:
            xs = np.hstack((states, actions))
        else:
            # giving state
            xs = states
        self._include_action = include_action
        self._include_action_to_basis = include_action_to_basis

        scaler = sklearn.preprocessing.StandardScaler()

        scaler.fit(xs)
        self._scaler = scaler
        self._n_component = n_component
        self._phi = self.fit(scaler.transform(xs))


    def fit(self, scaled):
        phi = sklearn.pipeline.FeatureUnion([
                ("rbf1", RBFSampler(gamma=5.0, n_components=self._n_component)),
                ("rbf2", RBFSampler(gamma=2.0, n_components=self._n_component)),
                ("rbf3", RBFSampler(gamma=1.0, n_components=self._n_component)),
                ("rbf4", RBFSampler(gamma=0.5, n_components=self._n_component))
                ])
        phi.fit(scaled)
        return phi


    def transform(self, s, a):
        """
        """
        # giving state action
        if self._include_action_to_basis:
            sa = np.hstack((s, a))
            if len(sa.shape) == 1:
                sa = np.expand_dims(sa, axis=0)
            x = self._scaler.transform(sa)
            featurized = self._phi.transform(x)
            return featurized
        elif self._include_action:
            if len(s.shape) == 1:
                s = np.expand_dims(s, axis=0)
            x = self._scaler.transform(s)
            featurized = self._phi.transform(x)
            return np.expand_dims(np.hstack((featurized[0], a)), axis=0)
        else:
            if len(s.shape) == 1:
                s = np.expand_dims(s, axis=0)
            x = self._scaler.transform(s)
            featurized = self._phi.transform(x)
            return featurized

        # giving s, a, b
        #ones = np.ones((featurized.shape[0], 1))
        #return np.hstack((featurized, ones))
        #ones = np.ones((featurized.shape[0], 1))
        #return np.hstack((featurized, ones))


def get_rbf_basis(env, n_component=25, include_action_to_basis=False, include_action=False):
    return RBFKernel(env, n_component=n_component, include_action_to_basis=include_action_to_basis, include_action=include_action).transform


def get_linear_basis():
    def f(s, a):
        sa = np.hstack((s, a))
        if len(sa.shape) == 1:
            sa = np.expand_dims(sa, axis=0)
        return sa
    return f


