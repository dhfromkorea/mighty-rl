class MDP(object):

    """Docstring for MDP. """

    def __init__(self, env):
        """TODO: to be defined1.

        Parameters
        ----------
        env : TODO

        """
        self._env = env
        # @todo: make this flexible
        self._s = env.reset()


    def step(self, a):
        """TODO: Docstring for function.

        Parameters
        ----------
        arg1 : action

        Returns
        -------
        TODO

        """
        return env.step(a)


    def reset(self):
        """TODO: Docstring for reset.

        Parameters
        ----------
        arg1 : TODO

        Returns
        -------
        TODO

        """
        s0 = env.reset()
        self._s = s0
        return s0




class MDPR(MDP):

    """Docstring for MDP. """

    def __init__(self, env, T, R):
        """TODO: to be defined1.

        Parameters
        ----------
        env : TODO
        T : TODO
        R : TODO

        """
        self._env = env
        self._T = T
        self._R = R
        # @todo: make this flexible
        self._s = env.reset()


    def step(self, a):
        """TODO: Docstring for function.

        Parameters
        ----------
        arg1 : action

        Returns
        -------
        TODO

        """
        s_next, r, done, _ =  env.step(a)
        return s_next, self._R(s, a), done, _


    def reset(self):
        """TODO: Docstring for reset.

        Parameters
        ----------
        arg1 : TODO

        Returns
        -------
        TODO

        """
        s0 = env.reset()
        self._s = s0
        return s0
