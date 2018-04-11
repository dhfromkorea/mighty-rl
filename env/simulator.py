from collections import namedtuple
from itertools import count
import sys
import numpy as np
import logging

from util import plotting

T = namedtuple("Transition", ["s", "a", "r", "s_next", "absorb"])


class Simulator(object):
    """Docstring for MountainCar. """

    def __init__(self, env, state_dim, action_dim):
        """TODO: to be defined1.

        Parameters
        ----------
        n_trials : TODO
        n_episodes : TODO
        max_iter : TODO
        gamma : TODO
        alpha : TODO


        """
        self._env = env
        self._state_dim = state_dim
        self._action_dim = action_dim


    def simulate(self, pi, n_trial, n_episode, return_stats=False, reward_fn=None):
        """TODO: Docstring for simulate

        Parameters
        ----------
        pi : behavior policy

        Returns
        -------
        D: a collection of transition samples

        """
        #@todo: remove the hard-coded max iter of 200 for mountaincar
        max_iter = 200

        stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(n_episode),
            episode_rewards=np.zeros(n_episode))

        D = []

        env = self._env
        for trial_i in range(n_trial):
            #D_t = D[trial_i]

            for epi_i in range(n_episode):

                last_reward = stats.episode_rewards[epi_i - 1]
                sys.stdout.flush()
                #D_e = D_t[epi_i]
                traj = []
                s = env.reset()

                for t in count():
                    a = pi.choose_action(s)
                    s_next, r, done, _ = env.step(a)

                    if reward_fn is not None:
                        r = reward_fn(s)


                    absorb = done and (t + 1) < max_iter

                    stats.episode_rewards[epi_i] += r
                    stats.episode_lengths[epi_i] = t

                    logging.debug("s {} a {} s_next {} r {} absorb {}".format(s, a, r, s_next,
                        absorb))

                    transition = T(s=s, a=a, r=r, s_next=s_next, absorb=absorb)
                    traj.append(transition)

                    if done:
                        logging.info("done after {} steps".format(t))
                        break

                    s = s_next


                    print("\rStep {} @ Episode {}/{} ({})".format(t, epi_i + 1, n_episode, last_reward), end="")

                D.append(traj)
        if return_stats:
            return D, stats
        else:
            return D

    def simulate_mixed(self, env, pi_list, sample_size, mix_ratio):
        """TODO: Docstring for simulate_mixed.

        Parameters
        ----------
        env : TODO
        pi_list : TODO
        sample_size : TODO
        mix_ratio : TODO

        Returns
        -------
        TODO

        """
        traj_list = []
        for pi, r in zip(pi_list, mix_ratio):
            trajs = self.simulate(pi, n_trial=1, n_episode=int(r * sample_size))
            traj_list += trajs
        return traj_list


    @staticmethod
    def to_matrix(D):
        """TODO: Docstring for process_.

        Parameters
        ----------
        o_ma : TODO

        Returns
        -------
        TODO

        """

        D_ = np.empty((0, 5))
        for traj in D:
            D_ = np.vstack((D_, np.array(traj)))
        return D_




