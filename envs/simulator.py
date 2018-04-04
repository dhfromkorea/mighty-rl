from collections import namedtuple
from itertools import count

import numpy as np
import logging
logging.basicConfig(filename="debug.log", level=logging.DEBUG)

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


    def simulate(self, pi, n_trial, n_episode):
        """TODO: Docstring for simulate

        Parameters
        ----------
        pi : behavior policy

        Returns
        -------
        D: a collection of transition samples

        """
        #D = [[[] for _ in range(self._n_episode)] for _ in range(self._n_trial)]
        D = []

        env = self._env
        for trial_i in range(n_trial):
            #D_t = D[trial_i]

            for epi_i in range(n_episode):

                #D_e = D_t[epi_i]
                traj = []
                s = env.reset()

                for t in count():
                    a = pi.choose_action(s)
                    s_next, r, done, _ = env.step(a)
                    logging.debug("s {} a {} s_next {} r {} done {}".format(s, a, r, s_next, done))
                    transition = T(s=s, a=a, r=r, s_next=s_next, done=done)
                    traj.append(transition)

                    s = s_next
                    if done:
                        logging.debug("done after {} steps".format(t))
                        break

                D.append(traj)
        return D


