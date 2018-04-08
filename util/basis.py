import numpy as np
import sklearn.pipeline
from sklearn.kernel_approximation import RBFSampler
import sklearn.preprocessing


class RBFKernel(object):

    """Docstring for RBFKernel. """

    def __init__(self, env, n_component=25):
        """TODO: to be defined1.

        Parameters
        ----------
        env : TODO
        n_component : TODO, optional


        """
        states = np.array([env.observation_space.sample() for x in range(10000)])
        #actions = np.array([env.action_space.sample() for x in range(10000)]).reshape(10000, 1)
        #xs = np.hstack((states, actions))

        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(states)
        self._scaler = scaler
        self._n_component = n_component
        self._phi = self.fit(scaler.transform(states))


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
        """"""
    #    sa = np.hstack((s, a))
    #    if len(sa.shape) == 1:
    #        sa = np.expand_dims(sa, axis=0)
    #    x = self._scaler.transform(sa)
        if len(s.shape) == 1:
            s = np.expand_dims(s, axis=0)
        x = self._scaler.transform(s)
        featurized = self._phi.transform(x)
        # @hack: hardcode action to be zero to ignore actions
        # for mountaincar-v0, basis features only on state is enough
        a = 0
        return np.hstack((featurized[0], a))


def get_rbf_basis(env, n_component=25):
    # Feature Preprocessing: Normalize to zero mean and unit variance
    # We use a few samples from the observation space to do this
    states = np.array([env.observation_space.sample() for x in range(10000)])
    actions = np.array([env.action_space.sample() for x in range(10000)]).reshape(10000, 1)
    xs = np.hstack((states, actions))

    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(xs)

    phi_rbf = get_phi(scaler, scaled=scaler.transform(xs), n_component=n_component)
    return phi_rbf


def get_phi(scaler, scaled, n_component):
    # Used to converte a state to a featurizes represenation.
    # We use RBF kernels with different variances to cover different parts of the space
    phi = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=n_component)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=n_component)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=n_component)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=n_component))
            ])
    phi.fit(scaled)

    def f(s, a):
        sa = np.hstack((s, a))
        if len(sa.shape) == 1:
            sa = np.expand_dims(sa, axis=0)
        x = scaler.transform(sa)
        return phi.transform(x)
    return f

def get_psi(scaler, scaled):
    # Used to converte a state to a featurizes represenation.
    # We use RBF kernels with different variances to cover different parts of the space
    psi = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=25)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=25)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=25)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=25))
            ])
    psi.fit(scaled)

    def f(s, a):
        sa = np.hstack((s, a))
        if len(sa.shape) == 1:
            sa = np.expand_dims(sa, axis=0)
        x = scaler.transform(sa)
        return psi.transform(x)

    return f

def simple_phi(s, a):
    # identity
    try:
        sa = np.hstack((s, a))
        if len(sa.shape) == 1:
            sa = np.expand_dims(sa, axis=0)
    except:
        import pdb;pdb.set_trace()
    return sa
    #return np.expand_dims(np.hstack((s, a)), axis=1)


