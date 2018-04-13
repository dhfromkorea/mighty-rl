import numpy as np
import sklearn.pipeline
from sklearn.kernel_approximation import RBFSampler
import sklearn.preprocessing


class RBFKernel(object):
    """Docstring for RBFKernel. """

    def __init__(self, env, n_component=25, gammas=[1.0], include_action_to_basis=False, include_action=False):
        """TODO: to be defined1.

        Parameters
        ----------
        env : TODO
        n_component : TODO, optional
        """
        # todo from D
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
        self._gammas = gammas
        self._phi = self.fit(scaler.transform(xs))


    def fit(self, scaled):
        feature_list = []
        for i, g in enumerate(self._gammas):
            f = ("rbf{}".format(i), RBFSampler(gamma=g, n_components=self._n_component))
            feature_list.append(f)
        phi = sklearn.pipeline.FeatureUnion(feature_list)
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


class RBFKernel2(object):
    """Docstring for RBFKernel. """

    def __init__(self, env, n_action, p, n_component=25, gammas=[1.0], include_action=False):
        """TODO: to be defined1.

        assume action is discrete

        Returns: n x (|A| x k) feature matrix

        Parameters
        ----------
        env : TODO
        n_component : TODO, optional
        """
        # todo from D
        states = np.array([env.observation_space.sample() for x in range(10000)])
        # giving state action
        self._include_action = include_action

        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(states)
        self._scaler = scaler

        self._n_component = n_component
        self._gammas = gammas
        self._n_action = n_action
        self._p = p
        self._phi = self.fit(scaler.transform(states))


    def fit(self, scaled):
        feature_list = []
        for i, g in enumerate(self._gammas):
            f = ("rbf{}".format(i), RBFSampler(gamma=g, n_components=self._n_component))
            feature_list.append(f)
        phi = sklearn.pipeline.FeatureUnion(feature_list)
        phi.fit(scaled)
        return phi


    def transform(self, s, a):
        """
        """
        def helper(s, a):
            featurized = np.zeros(self._n_action * self._p, dtype=np.float)
            featurized[self._p*a:self._p*(a+1)] = np.array(s)
            return featurized

        def helper_batch(x):
            phi_s, a = x[:-1], x[-1]
            a = a.astype(np.int)
            featurized = np.zeros(self._n_action * self._p, dtype=np.float)
            featurized[self._p*a:self._p*(a+1)] = phi_s
            return featurized

        if len(s.shape) == 1:
            s = np.expand_dims(s, axis=0)
        s = self._scaler.transform(s)
        phi_s = self._phi.transform(s)

        if s.shape[0] == 1:
            phi_sa = np.expand_dims(helper(phi_s, a), axis=0)
        else:
            phi_sa = np.apply_along_axis(helper_batch, 1, np.hstack((phi_s, a)))

        assert phi_sa.shape == (s.shape[0], self._n_action * self._p)
        #print("phi_sa:{}".format(phi_sa[0, :, :]))
        return phi_sa


        # giving state action
        if len(s.shape) == 1:
            s = np.expand_dims(s, axis=0)
        x = self._scaler.transform(s)
        featurized = self._phi.transform(x)
        return np.expand_dims(np.hstack((featurized[0], a)), axis=0)



def get_rbf_basis(env, n_component=25, gammas=[1.0], include_action_to_basis=False, include_action=False):
    return RBFKernel(env, n_component=n_component, gammas=gammas, include_action_to_basis=include_action_to_basis, include_action=include_action).transform


def get_linear_basis(include_action=False):
    def f(s, a):
        if include_action:
            sa = np.hstack((s, a))
            if len(sa.shape) == 1:
                sa = np.expand_dims(sa, axis=0)
            return sa
        else:
            if len(s.shape) == 1:
                s = np.expand_dims(s, axis=0)
            return s
    return f



def get_linear_basis2(p, n_action, include_action=False):
    """
    assume action is discrete

    Returns: n x (|A| x k) feature matrix
    """
    def helper(s, a):
        featurized = np.zeros(n_action * p, dtype=np.float)
        featurized[p*a:p*(a+1)] = np.array(s)
        return featurized

    def f(s, a):
        if len(s.shape) == 1:
            s = np.expand_dims(s, axis=0)
        phi = np.apply_along_axis(helper, s, a)
        import pdb;pdb.set_trace()
        assert phi.shape == (s.shape[0], n_action, p)
        return phi
    return f


