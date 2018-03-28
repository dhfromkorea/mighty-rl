import numpy as np
from tqdm import tqdm
from mdp.solver import Q_value_iteration
from policy.custom_policy import get_physician_policy
from policy.policy import GreedyPolicy, RandomPolicy, StochasticPolicy
from irl.irl import *
from optimize.quad_opt import QuadOpt
from constants import NUM_STATES, NUM_ACTIONS, DATA_PATH
from optimize.mmpsolver import MMTOpt


def run_maximum_margin_planning(transition_matrix_train,
                            transition_matrix,
                            reward_matrix,
                            pi_expert,
                            initial_state_probs,
                            phi,
                            num_exp_trajectories,
                            svm_penalty,
                            svm_epsilon,
                            num_iterations,
                            num_trials,
                            use_stochastic_policy,
                            features,
                            state_action_histogram,
                            slack_scale,
                            alpha,
                            use_slack,
                            theta,
                            gamma,
                            verbose):

    mu_expert, v_expert = estimate_feature_expectation(transition_matrix,
                                                             initial_state_probs,
                                                             phi,
                                                             pi_expert,
                                                             num_trajectories=num_exp_trajectories)
    if verbose:
        print('objective: get close to ->')
        print('avg mu_expert', np.mean(mu_expert))
        print('v_expert', v_expert)
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
    state_histogram = np.sum(state_action_histogram, axis=1)
    # normalize
    state_histogram /= np.sum(state_action_histogram)

    for trial_i in tqdm(range(num_trials)):
        if verbose:
            print('max margin IRL starting ... with {}th trial'.format(1+trial_i))

        # step 1: initialize pi_irl and mu_irl
        pi_irl = RandomPolicy(NUM_PURE_STATES, NUM_ACTIONS)
        mu_irl, v_irl = estimate_feature_expectation(transition_matrix_train,
                                                           initial_state_probs,
                                                           phi, pi_irl)


        '''
            min w.r.t. W, Z
            (1/2) ||w||^2 + slack_scale * ||s||_slack_norm
            s.t w.mu(expert) >= w.mu(candidate) + alpha * loss(candidate) - s forall candidates
        '''
        opt = MMTOpt(pi_expert=pi_expert,
                     slack_scale = slack_scale,
                     alpha = alpha,
                     use_slack = use_slack,
                     state_histogram=state_histogram,
                     slack_norm=2)

        best_actions_old = None
        W_old = None
        pi_irls = np.array([None]*num_iterations)
        weights = np.array([None]*num_iterations)
        for i in range(num_iterations):
            # step 2: solve qp
            res = opt.optimize(mu_expert=mu_expert,
                                     mu_irl=mu_irl,
                                     pi_irl=pi_irl)
            W, s, margin = res['w'], res['s'], res['margin']
            if verbose:
                print ("i in num_iterations", i)
                print('weights', W)
                print('slack', s)
                print('margin', margin)
            weights[i] = W
            # step 4: solve mdpr
            compute_reward = make_reward_computer(W, phi)
            reward_matrix = np.asarray([compute_reward(s) for s in range(NUM_STATES)])
            Q_star = Q_value_iteration(transition_matrix_train, reward_matrix, theta, gamma, NUM_TERMINAL_STATES)
            if use_stochastic_policy:
                pi_irl = StochasticPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q_star)
            else:
                pi_irl = GreedyPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q_star)
            pi_irls[i] = pi_irl
            # step 5: estimate mu pi tilda
            mu_irl, v_irl = estimate_feature_expectation(
                                   transition_matrix_train,
                                   initial_state_probs,
                                   phi, pi_irl)
            dist_mu = np.linalg.norm(mu_irl - mu_expert, 2)
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
                print('avg mu_irl', np.mean(mu_irl))
                print('dist_mu', dist_mu)
                print('margin', margin)
                print('v_pi', v_irl)
                print('')

            # step 6: saving plotting vars
            dist_mus[trial_i, i] = dist_mu
            margins[trial_i, i] = margin
            v_pis[trial_i, i] = v_irl
            intermediate_reward_matrix += reward_matrix
        # find a near-optimal policy from a policy reservoir
        # taken from Abbeel (2004)
        # TODO: retrieve near-optimal expert policy
        min_margin_iter_idx = np.argmin(margins[trial_i])
        approx_exp_weights[trial_i] = weights[min_margin_iter_idx]
        approx_exp_policies[trial_i] = pi_irls[min_margin_iter_idx].Q

        if verbose:
            print('best weights at {}th trial'.format(trial_i), weights[min_margin_iter_idx])
            print('best Q at {}th trial'.format(trial_i), pi_irls[min_margin_iter_idx].Q)

    # there will be a better way to do a policy selection
    approx_expert_weights = np.mean(approx_exp_weights, axis=0)
    compute_reward = make_reward_computer(approx_expert_weights, phi)
    intermediate_reward_matrix = np.asarray([compute_reward(s) for s in range(NUM_PURE_STATES)])
    approx_expert_Q = np.mean(approx_exp_policies, axis=0)

    if use_stochastic_policy:
        pi_irl = StochasticPolicy(NUM_PURE_STATES, NUM_ACTIONS, approx_expert_Q)
    else:
        pi_irl = GreedyPolicy(NUM_PURE_STATES, NUM_ACTIONS, approx_expert_Q)
    feature_importances = sorted(zip(features, approx_expert_weights), key=lambda x: x[1], reverse=True)
    results = {'margins': margins,
               'dist_mus': dist_mus,
               'v_pis': v_pis,
               'v_pi_expert': v_expert,
               'intermediate_rewards': intermediate_reward_matrix,
               'approx_expert_weights': approx_expert_weights,
               'feature_imp': feature_importances,
               'num_exp_trajectories': num_exp_trajectories,
               'approx_expert_Q': approx_expert_Q,
               'approx_expert_policy': pi_irl
              }
    return results

