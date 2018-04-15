import itertools
import numpy as np
import tensorflow as tf

import baselines
from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import BatchInput
from baselines.common.schedules import LinearSchedule
import baselines.common.tf_util as U


import os
path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(path, os.pardir))

import logging

class DQN(object):
    """Docstring for DQN.


    TODO:
        check the perf of the basic model
        consider some additional architecture
        https://github.com/vmayoral/basic_reinforcement_learning/blob/master/tutorial5/dqn-mountaincar.py
        https://gist.github.com/avalcarce/93991f052ecbf19cfef99c76b8f0b470


    """

    def __init__(self, env,
                       D=None,
                       model=None,
                       hiddens=[64],
                       learning_rate=1e-3,
                       gamma=0.99,
                       buffer_size=50000,
                       max_timesteps=10**6,
                       print_freq=10,
                       layer_norm=True,
                       exploration_fraction=0.1,
                       exploration_initial_eps=1.0,
                       exploration_final_eps=0.1,
                       target_network_update_freq=500,
                       policy_evaluate_freq=5000,
                       param_noise=True,
                       grad_norm_clipping=10,
                       buffer_batch_size=32):
        """TODO: to be defined1.

        Parameters
        ----------
        env : TODO
        D : TODO, optional


        """
        if model is None:
            self._model = deepq.models.mlp(hiddens, layer_norm=layer_norm)
        else:
            self._model = model
        self._env = env
        self._D = D

        # for batch
        # Create the replay buffer
        # PER?
        logging.info("dumping D of size {} into experience replay".format(D.shape))
        n_sample = D.shape[0]
        self._replay_buffer = ReplayBuffer(n_sample)
        for episode in D:
            s, a, r, s_next, _, done = episode
            self._replay_buffer.add(s, a, r, s_next, float(done))


        self._hiddens = hiddens
        self._lr = learning_rate
        self._gamma = gamma
        self._buffer_size = buffer_size
        self._max_timesteps = max_timesteps
        self._print_freq = print_freq
        self._exploration_fraction = exploration_fraction
        self._exploration_initial_eps = exploration_initial_eps
        self._exploration_final_eps = exploration_final_eps
        self._param_noise = param_noise
        self._layer_norm = layer_norm
        self._grad_norm_clipping = grad_norm_clipping
        self._buffer_batch_size = buffer_batch_size
        self._target_network_update_freq = target_network_update_freq
        self._policy_evaluate_freq = policy_evaluate_freq


    def train(self, use_batch=False):
        """TODO: Docstring for train.

        Parameters
        ----------
        use_batch : TODO

        Returns
        -------
        TODO

        """
        if use_batch:
            return self._train_batch()
        else:
            return self._train_online()


    def _train_online(self):
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
        # consider changing this
        act = deepq.learn(
                        self._env,
                        q_func=self._model,
                        lr=self._lr,
                        gamma=self._gamma,
                        max_timesteps=self._max_timesteps,
                        buffer_size=self._buffer_size,
                        exploration_fraction=self._exploration_fraction,
                        exploration_final_eps=self._exploration_final_eps,
                        print_freq=self._print_freq,
                        param_noise=self._param_noise,
                        target_network_update_freq=self._target_network_update_freq
                    )
        print("Saving model to mountaincar_model.pkl")
        act.save("{}/data/mountaincar_model.pkl".format(root_path))
        self._policy = act
        return act



    def _train_batch(self):
        """TODO: Docstring for _train_online

        perform mini-batch GD with samples from D
        first put D in experience replay buffer

        Parameters
        ----------
        arg1 : TODO

        Returns
        -------
        TODO

        """

        sess = tf.Session()
        sess.__enter__()

        def make_obs_ph(name):
            return BatchInput(self._env.observation_space.shape, name=name)

        tools = deepq.build_train(
            make_obs_ph=make_obs_ph,
            q_func=self._model,
            num_actions=self._env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=self._lr),
            gamma=self._gamma,
            grad_norm_clipping=self._grad_norm_clipping,
        )
        act, train, update_target, debug = tools


        # dump everything to experience replay
        # at the initialization of this class

        # create exploration schedule (not needed for batch)
        self._timestep = int(self._exploration_fraction * self._max_timesteps),

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        # Iterate through episodes, adding the data to replay buffer

        #num_epochs = 100
        for t in itertools.count():
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            s, a, r, s_next, dones = self._replay_buffer.sample(self._buffer_batch_size)
            td_errors = train(s, a, r, s_next, dones, np.ones_like(r))
            if t % self._target_network_update_freq == 0:
                logging.info("been trained {} steps".format(t))
                update_target()
            if t > 100 and t % self._policy_evaluate_freq == 0:
                logging.info("evaluating the policy...")
                self._evaluate_policy(act)


        self._policy = act
        print("Saving model to mountaincar_model.pkl")
        act.save("{}/data/mountaincar_model.pkl".format(root_path))
        return act


    def _evaluate_policy(self, act, max_iter=3000):
        """TODO: Docstring for evalute_policy.

        Parameters
        ----------
        act : TODO

        Returns
        -------
        TODO

        """
        episode_rewards = [0.0]
        obs = self._env.reset()
        for t in itertools.count():
            if t > max_iter:
                break
            # Take action and update exploration to the newest value
            #action = act(obs[None], update_eps=exploration.value(t))[0]
            action = act(obs[None])[0]
            new_obs, rew, done, _ = self._env.step(action)
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = self._env.reset()
                # needed?
                episode_rewards.append(0)


        logger.record_tabular("steps", t)
        logger.record_tabular("episodes", len(episode_rewards))
        logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
        logger.record_tabular("% time spent exploring", int(100 * 0.0))
        logger.dump_tabular()

