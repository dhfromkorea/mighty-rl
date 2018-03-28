import numpy as np
from tqdm import tqdm
from mdp.solver import Q_value_iteration
from policy.custom_policy import get_physician_policy
from policy.policy import GreedyPolicy, RandomPolicy, StochasticPolicy
from irl.irl import *
from optimize.quad_opt import QuadOpt
from constants import NUM_STATES, NUM_ACTIONS, DATA_PATH

import logging
logging.basicConfig(filename="debug.log", level=logging.DEBUG)

class BatchApprenticeshipLearning(object):
    """Batch ApprenticeshipLearning continuous state"""

    def __init__(self, env,
                       pi_init,
                       D,
                       p,
                       q,
                       phi,
                       psi,
                       gamma,
                       eps,
                       mu_exp,
                       init_s_sampler,
                       mu_sample_size):
        """TODO: to be defined1.

        Parameters
        ----------
        env : TODO
        pi_init : TODO

        D : trajectory data
        p : dimension of phi
        q : dimension of psi
        phi : basis function for reward
        psi : basis function for feature expectation
        gamma : discount factor (0, 1)
        eps : small positive value to make A invertible
        mu_exp : TODO
        init_s_sampler: initial state sampler
        mu_sample_size : sample size to account for varying init states
        """
        self._env = env
        self._pi_init = pi_init

        self._D = D
        self._p = p
        self._q = q
        self._phi = phi
        self._psi = psi
        self._gamma = gamma
        self._eps = eps
        self._mu_exp = mu_exp
        self._init_s_sampler = init_s_sampler
        self._mu_sample_size = mu_sample_size


    def run(self, n_trial, n_iteration):
        """TODO: Docstring for something.

        Parameters
        ----------
        n_trial : TODO
        n_iteration : TODO
        D : batch data under pi_expert

        Returns
        -------
        TODO
        - check if we should use MC estimator for mu_exp or lstd
        - check if to marginalize action out from mu estimator
        - check how to sample initial state
        - check scaler
        """
        mu_exp = self._mu_exp

        # todo: traj legnth or value of pi_expert
        margin_v_collection = []
        margin_mu_collection = []
        pi_collection = []
        weight_collection = []
        mu_collection = []
        pi_best_collection = []

        for trial_i in tqdm(range(n_trial)):
            margin_v_list = []
            margin_mu_list = []
            pi_list = []
            weight_list = []
            mu_list = []

            # assuming pi_init is deterministic
            # otherwise we need to integrate out
            mu_irl = self.estimate_mu(self.pi_init)

            # todo replace with cvxpy
            opt = QuadOpt(epsilon=0.0001,
                          penalty=0.00001)
            for epi_i in range(n_episode):
                W, converged, margin_v, margin_mu = opt.optimize(mu_exp, mu_irl)
                # record margin_v, margin_mu
                weight_list.append(W)
                margin_v_list.append(margin_v)
                margin_mu_list.append(margin_mu)

                if converged:
                    logging.info("margin_mu converged")

                get_reward = get_reward_fn(W=W)

                # construct a new mdp with the reward

                # solve the mdp
                pi_irl = mdp_solver.solver(mdp)
                pi_list.append(pi_irl)

                # record new mu_irl
                mu_irl = self.estimate_mu(pi_irl)
                mu_list.append(mu_irl)

            # save trial-level data
            margin_v_collection.append(margin_v_list)
            margin_mu_collection.append(margin_mu_list)
            pi_collection.append(pi_list)
            weight_collection.append(weight_list)
            mu_collection.append(mu_list)


            # choose the best policy for each trial
            pi_best = self.choose_pi_best(pi_list)
            pi_best_collection.append(pi_best)

        # dump save the important meta data to numpy
        results = {}
        return results

    def choose_pi_best(self, pi_list):
        """TODO: Docstring for choose_pi_best.

        Parameters
        ----------
        pi_list : TODO

        Returns
        -------
        pi_best

        """
        pass


    def estimate_mu(self, pi_eval):
        """TODO: Docstring for something.

        need to refit using a new policy to evaluate
        Parameters
        ----------
        pi_eval : policy under which to estimate mu

        Returns
        -------
        TODO

        """
        mu_estimator = LSTDMu(p=self.p,
                              q=self.q,
                              phi=self.phi,
                              psi=self.psi,
                              gamma=self.gamma,
                              eps=self.eps)

        mu_estimator.fit(D=self._D, pi=pi_eval)

        init_state_list = [self._init_s_sampler() for _ in range(self._mu_sample_size)]
        mu_list = []

        for s in init_state_list:
            mu = mu_estimator.predict(s, pi_eval(s))
            mu_list.append(mu)

        mu_hat = np.array(mu_list).mean(axis=0)
        return mu_hat


    def get_reward_fn(self, W):
        """linearly parametrize reward function.

        Parameters
        ----------
        W : weight

        Returns
        -------
        TODO
        - think whether to do s, a or just s

        """
        #return lambda s : W.dot(self.phi(s))
        return lambda s, a : W.dot(self.phi(s, a))





def run_apprenticeship_learning(transition_matrix_train,
                    transition_matrix,
                    reward_matrix,
                    pi_expert,
                    initial_state_probs,
                    phi,
                    mu_sample_size,
                    theta,
                    gamma,
                    svm_penalty,
                    svm_epsilon,
                    num_iterations,
                    num_trials,
                    use_stochastic_policy,
                    features,
                    hyperplane_margin,
                    verbose):

    '''
    reproduced apprenticeship learning algorithm
    described in Apprenticeship Learning paper (Abbeel and Ng, 2002)
    with Quadratic Programming
    returns:
    results = {'margins': margins,
               'dist_mus': dist_mus,
               'v_pis': v_pis,
               'v_pi_expert': v_pi_expert,
               'svm_penlaty': svm_penalty,
               'svm_epsilon': svm_epsilon,
               'approx_expert_weights': approx_expert_weights,
               'num_exp_trajectories': num_exp_trajectories,
               'approx_expert_Q': approx_expert_Q
              }
    it is important we use only transition_matrix_train for training
    when testing, we will use transition_matrix, which is a better approximation of the world
    '''
    # some hardcoded hyperparams
    # potential args
    max_iter = 25

    mu_pi_expert, v_pi_expert, traj_len_expert = estimate_feature_expectation(transition_matrix,
                                                             initial_state_probs,
                                                             phi,
                                                             pi_expert,
                                                             sample_size=mu_sample_size,
                                                             gamma=gamma,
                                                             max_iter=max_iter)
    mus = {}
    mus['expert'] = mu_pi_expert
    mus['irl'] = [[] for _ in range(num_trials)]

    if verbose:
        print('objective: get close to ->')
        print('avg mu_pi_expert', np.mean(mu_pi_expert))
        print('v_pi_expert', v_pi_expert)
        print('trajectory length expert', traj_len_expert)
        print('')

    # initialize vars for plotting
    # initialize this with 10000.0 because if may converge
    # in the middle
    margins = np.full((num_trials, num_iterations), 10000.0)
    dist_mus = np.full((num_trials, num_iterations), 10000.0)
    v_pis = np.zeros((num_trials, num_iterations))
    intermediate_reward_matrix = np.zeros((reward_matrix.shape))
    approx_exp_policies = np.array([None] * num_trials)
    approx_exp_weights = np.array([None] * num_trials)


    for trial_i in tqdm(range(num_trials)):
        if verbose:
            print('max margin IRL starting ... with {}th trial'.format(1+trial_i))

        # step 1: initialize pi_tilda and mu_pi_tilda
        pi_tilda = RandomPolicy(NUM_PURE_STATES, NUM_ACTIONS)
        mu_pi_tilda, v_pi_tilda, traj_len_tilda = estimate_feature_expectation(transition_matrix_train,
                                                           initial_state_probs,
                                                           phi,
                                                           pi_tilda,
                                                           sample_size=mu_sample_size,
                                                           gamma=gamma,
                                                           max_iter=max_iter)
        opt = QuadOpt(epsilon=svm_epsilon,
                      penalty=svm_penalty,
                      hyperplane_margin=hyperplane_margin)
        best_actions_old = None
        W_old = None
        pi_tildas = np.array([None]*num_iterations)
        weights = np.array([None]*num_iterations)
        for i in range(num_iterations):
            # step 2: solve qp
            W, converged, margin = opt.optimize(mu_pi_expert, mu_pi_tilda)
            # step 3: terminate if margin <= epsilon
            if converged:
                print('margin coverged with', margin)
                break

            weights[i] = W
            # step 4: solve mdpr
            compute_reward = make_reward_computer(W, phi)
            reward_matrix = np.asarray([compute_reward(s) for s in range(NUM_STATES)])
            Q_star = Q_value_iteration(transition_matrix_train, reward_matrix, theta, gamma, NUM_TERMINAL_STATES)
            if use_stochastic_policy:
                pi_tilda = StochasticPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q_star)
            else:
                pi_tilda = GreedyPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q_star)
            pi_tildas[i] = pi_tilda
            # step 5: estimate mu pi tilda
            mu_pi_tilda, v_pi_tilda, traj_len_tilda = estimate_feature_expectation(transition_matrix_train,
                                                                                   initial_state_probs,
                                                                                   phi,
                                                                                   pi_tilda,
                                                                                   sample_size=mu_sample_size,
                                                                                   gamma=gamma,
                                                                                   max_iter=max_iter)

            dist_mu = np.linalg.norm(mu_pi_tilda - mu_pi_expert, 2)
            if verbose:
                # intermediate reeport for debugging
                print('max intermediate rewards: ', np.max(reward_matrix[:-2]))
                print('avg intermediate rewards: ', np.mean(reward_matrix[:-2]))
                print('min intermediate rewards: ', np.min(reward_matrix[:-2]))
                print('sd intermediate rewards: ', np.std(reward_matrix[:-2]))
                best_actions = np.argmax(Q_star, axis=1)
                if best_actions_old is not None:
                    actions_diff = np.sum(best_actions != best_actions_old)
                    actions_diff /= best_actions.shape[0]
                    print('(approx.) argmax Q changed (%)', 100*actions_diff)
                    best_actions_old = best_actions

                if W_old is not None:
                    print('weight difference (l2 norm)', np.linalg.norm(W_old - W, 2))
                W_old = W
                print('avg mu_pi_tilda', np.mean(mu_pi_tilda))
                #print('dist_mu', dist_mu)
                print('margin', margin)
                print('v_pi', v_pi_tilda)
                print('trajectory length', traj_len_tilda)
                print('')

            # step 6: saving plotting vars
            dist_mus[trial_i, i] = dist_mu
            margins[trial_i, i] = margin
            v_pis[trial_i, i] = v_pi_tilda
            intermediate_reward_matrix += reward_matrix
            mus['irl'][trial_i].append(mu_pi_tilda)
        # find a near-optimal policy from a policy reservoir
        # taken from Abbeel (2004)
        # TODO: retrieve near-optimal expert policy
        min_margin_iter_idx = np.argmin(margins[trial_i])
        approx_exp_weights[trial_i] = weights[min_margin_iter_idx]
        approx_exp_policies[trial_i] = pi_tildas[min_margin_iter_idx].Q

        if verbose:
            print('best weights at {}th trial'.format(trial_i), weights[min_margin_iter_idx])
            print('best Q at {}th trial'.format(trial_i), pi_tildas[min_margin_iter_idx].Q)

    # there will be a better way to do a policy selection
    approx_expert_weights = np.mean(approx_exp_weights, axis=0)
    compute_reward = make_reward_computer(approx_expert_weights, phi)
    intermediate_reward_matrix = np.asarray([compute_reward(s) for s in range(NUM_PURE_STATES)])
    approx_expert_Q = np.mean(approx_exp_policies, axis=0)

    if use_stochastic_policy:
        pi_tilda = StochasticPolicy(NUM_PURE_STATES, NUM_ACTIONS, approx_expert_Q)
    else:
        pi_tilda = GreedyPolicy(NUM_PURE_STATES, NUM_ACTIONS, approx_expert_Q)
    feature_importances = sorted(zip(features, approx_expert_weights), key=lambda x: x[1], reverse=True)
    results = {'margins': margins,
               'dist_mus': dist_mus,
               'mus': mus,
               'v_pis': v_pis,
               'v_pi_expert': v_pi_expert,
               'svm_penlaty': svm_penalty,
               'svm_epsilon': svm_epsilon,
               'intermediate_rewards': intermediate_reward_matrix,
               'approx_expert_weights': approx_expert_weights,
               'feature_imp': feature_importances,
               'approx_expert_Q': approx_expert_Q,
               'approx_expert_policy': pi_tilda
              }
    return results
