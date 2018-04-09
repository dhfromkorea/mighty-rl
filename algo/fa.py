import gym
import itertools
import matplotlib
import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing

from util import plotting
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler


class Estimator():
    """
    Value Function approximator.
    """
    def __init__(self, env, phi):
        self._env = env
        self._phi = phi

        self.models = []
        # building a condition model
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            model.partial_fit([self._phi(env.reset(), 0)], [0])
            self.models.append(model)



    def predict(self, s, a=None):
        if a is None:
            Q = []
            for m, a in zip(self.models, range(self._env.action_space.n)):
                features = self._phi(s, a)
                Q.append(m.predict([features])[0])
            return np.array(Q)
        else:
            features = self._phi(s, a)
            return self.models[a].predict([features])[0]


    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        features = self._phi(s, a)
        self.models[a].partial_fit([features], [y])


# @todo: fix this
class EpsilonGreedyPolicy(object):
    def __init__(self, estimator, eps, nA):
        """TODO: to be defined1. """
        self._estimator = estimator
        self._eps = eps
        self._nA = nA

    def _action_probs(self, s):
        probs = np.ones(self._nA, dtype=float) * self._eps / self._nA
        q_values = self._estimator.predict(s)
        best_action = np.argmax(q_values)
        probs[best_action] += (1.0 - self._eps)
        return probs

    def choose_action(self, s):
        p = self._action_probs(s)
        a = np.random.choice(np.arange(len(p)), p=p)
        return a

    @property
    def coef_(self):
        """TODO: Docstring for function.

        Parameters
        ----------

        Returns
        -------
        coef : array
            dim_action by dim_feature array

        """
        # @todo: perhaps consider using one model for all actions
        return np.array([np.hstack((m.coef_, m.intercept_)) for m in self._estimator.models])



class LinearQ3(object):
    """Docstring for LinearQ3. """

    def __init__(self, env, phi, action_list, n_episode, epsilon, epsilon_decay, gamma):
        """TODO: to be defined1.

        Parameters
        ----------
        env : TODO
        phi : array
        action_list : list
        n_episode : TODO
        epsilon : TODO
        epsilon : TODO
        plotting : TODO
        gamma: float
            discount factor

        """
        self._env = env
        self._estimator = Estimator(env=env, phi=phi)
        self._n_episode = n_episode
        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._gamma = gamma


    def solve(self, reward_fn=None):
        """TODO: Docstring for solve.

        Parameters
        ----------
        reward_fn : function
            reward function for IRL

        Returns
        -------
        TODO

        """
        # Keeps track of useful statistics
        stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(self._n_episode),
            episode_rewards=np.zeros(self._n_episode))

        for i_episode in range(self._n_episode):

            pi = EpsilonGreedyPolicy(estimator=self._estimator,
                                     eps=self._epsilon * self._epsilon_decay**i_episode,
                                     nA=self._env.action_space.n)


            last_reward = stats.episode_rewards[i_episode - 1]
            sys.stdout.flush()

            state = self._env.reset()

            for t in itertools.count():


                action = pi.choose_action(state)

                next_state, reward, done, _ = self._env.step(action)

                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t

                q_values_next = self._estimator.predict(next_state)

                if reward_fn is not None:
                    reward_irl = np.asscalar(reward_fn(state, action))
                    td_target = reward_irl + self._gamma * np.max(q_values_next)
                else:
                    td_target = reward + self._gamma * np.max(q_values_next)


                self._estimator.update(state, action, td_target)


                print("\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1, self._n_episode, last_reward), end="")

                if done:
                    break

                state = next_state


        return pi, stats


if __name__ == "__main__":
    estimator = Estimator()

    # Note: For the Mountain Car we don't actually need an epsilon > 0.0
    # because our initial estimate for all states is too "optimistic" which leads
    # to the exploration of all states.
    stats = q_learning(env, estimator, 100, epsilon=0.0)

    plotting.plot_cost_to_go_mountain_car(env, estimator)
    plotting.plot_episode_stats(stats, smoothing_window=25)

