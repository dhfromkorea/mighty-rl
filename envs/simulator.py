from collections import namedtuple
from itertools import count
import sys
import numpy as np
import logging

import plotting

T = namedtuple("Transition", ["s", "a", "r", "s_next", "done"])


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


    def simulate(self, pi, n_trial, n_episode, return_stats=False):
        """TODO: Docstring for simulate

        Parameters
        ----------
        pi : behavior policy

        Returns
        -------
        D: a collection of transition samples

        """

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

                    stats.episode_rewards[epi_i] += r
                    stats.episode_lengths[epi_i] = t

                    logging.debug("s {} a {} s_next {} r {} done {}".format(s, a, r, s_next, done))
                    transition = T(s=s, a=a, r=r, s_next=s_next, done=done)
                    traj.append(transition)

                    s = s_next

                    if done:
                        logging.debug("done after {} steps".format(t))
                        break

                    print("\rStep {} @ Episode {}/{} ({})".format(t, epi_i + 1, n_episode, last_reward), end="")

                D.append(traj)
        if return_stats:
            return D, stats
        else:
            return D


