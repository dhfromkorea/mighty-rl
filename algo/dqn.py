import os
path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(path, os.pardir))

class DQN(object):

    """Docstring for DQN. """

    def __init__(self, env, D=None):
        """TODO: to be defined1.

        Parameters
        ----------
        env : TODO
        D : TODO, optional


        """
        self._env = env
        if D is not None:
            self._D = D


    def train(self, use_batch):
        """TODO: Docstring for train.

        Parameters
        ----------
        use_batch : TODO

        Returns
        -------
        TODO

        """
        pass


    def _train_online(self,
                      hiddens=[64],
                      learning_rate=1e-3,
                      buffer_size=50000,
                      max_timesteps=10**6,
                      print_freq=10,
                      exploration_fraction=0.1,
                      exploration_final_eps=0.1,
                      param_noise=True):
        """TODO: Docstring for _train_online.

        Parameters
        ----------
        print_freq : TODO, optional
        exploration_fraction : TODO, optional
        exploration_final_eps : TODO, optional

        Returns
        -------
        TODO

        """
        # Enabling layer_norm here is import for parameter space noise!
        self._model = deepq.models.mlp(hiddens, layer_norm=True)
        act = deepq.learn(
                        self._env,
                        q_func=self._model,
                        lr=learning_rate,
                        max_timesteps=max_timesteps,
                        buffer_size=buffer_size,
                        exploration_fraction=exploration_fraction,
                        exploration_final_eps=exploration_final_eps,
                        print_freq=print_freq,
                        param_noise=param_noise
                    )
        print("Saving model to mountaincar_model.pkl")
        act.save("{}/data/mountaincar_model.pkl".format(root_path))



    def _train_batch(self, arg1):
        """TODO: Docstring for _train_online.

        Parameters
        ----------
        arg1 : TODO

        Returns
        -------
        TODO

        """
        pass
