import numpy as np
from tqdm import tqdm
import cvxpy as cvx
import logging

from lstd import LSTDQ, LSTDMu, LSPI
from policy import LinearQ2

class ApprenticeshipLearning(object):
    """Docstring for ApprenticeshipLearning. """

    def __init__(self):
        """TODO: to be defined1. """
        pass


class BatchApprenticeshipLearning(ApprenticeshipLearning):
    """Batch ApprenticeshipLearning continuous state"""

    def __init__(self, env,
                       pi_init,
                       D,
                       action_list,
                       p,
                       q,
                       phi,
                       psi,
                       gamma,
                       eps,
                       mu_exp,
                       init_s_sampler,
                       mu_sample_size,
                       precision,
                       use_slack=False,
                       slack_penalty=0.01
                       ):
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
        precision : convergence threshold
        use_slack : whether to use slack for convex optimization
        slack_penalty : scaling term
        """
        self._env = env
        self._pi_init = pi_init

        self._D = D
        self._action_list = action_list
        self._p = p
        self._q = q
        self._phi = phi
        self._psi = psi
        self._gamma = gamma
        self._eps = eps
        self._mu_exp = mu_exp
        self._init_s_sampler = init_s_sampler
        self._mu_sample_size = mu_sample_size

        self._precision = precision
        self._use_slack = use_slack
        self._slack_penalty = slack_penalty


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

            # pick a random policy as init policy
            mu_irl = self.estimate_mu(self._pi_init)
            mu_list.append(mu_irl)
            pi_list.append(self._pi_init)

            # todo replace with cvxpy

            for epi_i in range(n_iteration):
                W, (margin_v, margin_mu, converged) = self._optimize(mu_list)

                # record margin_v, margin_mu
                weight_list.append(W)
                margin_v_list.append(margin_v)
                margin_mu_list.append(margin_mu)
                logging.info("W: {}".format(W))
                logging.info("margin_v: {}".format(margin_v))
                logging.info("margin_mu: {}".format(margin_mu))

                if converged:
                    logging.info("margin_mu converged after {} iterations".format(epi_i + 1))
                    break

                reward_fn = self._get_reward_fn(W=W)

                # @todo: allow non-batch solver also
                # solve the mdpr
                pi_irl = self.solve_mdpr(reward_fn)
                pi_list.append(pi_irl)
                logging.debug("pi_irl: {}".format(pi_irl._W))

                # record new mu_irl
                mu_irl = self.estimate_mu(pi_irl)
                mu_list.append(mu_irl)
                logging.debug("mu_irl: {}".format(mu_irl))

            # save trial-level data
            margin_v_collection.append(margin_v_list)
            margin_mu_collection.append(margin_mu_list)
            pi_collection.append(pi_list)
            weight_collection.append(weight_list)
            mu_collection.append(mu_list)


            # choose the best policy for each trial
            pi_best = self._choose_pi_best(mu_list, pi_list)
            pi_best_collection.append(pi_best)
            logging.info("pi_best: {}".format(pi_best._W))


        # dump save the important meta data to numpy
        results = {
                "margin_v": margin_v_collection,
                "margin_mu": margin_mu_collection,
                "mu": mu_collection,
                "weight": weight_collection,
                "policy": pi_collection,
                "solutions" : pi_best_collection,
                }
        return results


    def solve_mdpr(self, reward_fn):
        """TODO: Docstring for solve_mdpr

        note W here is a parameter for Q, not for reward
        assume D is good enough
        we should be able to find a good Q_hat

        Parameters
        ----------
        reward_fn : TODO

        Returns
        -------
        TODO

        """
        logging.info("solving MDP\R with LSPI")
        # modify reward
        # pi = phi(s,a)^T W_0
        # phi p x 1 theta p x 1
        np.random.seed(0)
        W_0 = np.random.rand(self._p)
        import time
        start = time.time()

        lspi = LSPI(D=self._D,
                    action_list=self._action_list,
                    p=self._p,
                    phi=self._phi,
                    gamma=self._gamma,
                    precision=self._precision,
                    eps=self._eps,
                    W_0=W_0,
                    reward_fn=reward_fn)

        print("lspi took", start - time.time())
        W = lspi.solve()
        pi = LinearQ2(action_list=self._action_list,
                      phi=self._phi,
                      W=W)
        return pi



    def _choose_pi_best(self, mu_list, pi_list):
        """TODO: Docstring for choose_pi_best.

        Parameters
        ----------
        pi_list : TODO

        Returns
        -------
        pi_best

        TODO
        - try solve for best convex combination to minimize mu distance

        """
        pi_best = pi_list[np.argmin(np.abs(self._mu_exp - np.array(mu_list)))]
        return pi_best


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
        mu_estimator = LSTDMu(p=self._p,
                              q=self._q,
                              phi=self._phi,
                              psi=self._psi,
                              gamma=self._gamma,
                              eps=self._eps)

        mu_estimator.fit(D=self._D, pi=pi_eval)

        init_state_list = [self._init_s_sampler() for _ in range(self._mu_sample_size)]
        mu_list = []

        for s in init_state_list:
            mu = mu_estimator.predict(s, pi_eval.choose_action(s))
            mu_list.append(mu)

        mu_hat = np.array(mu_list).mean(axis=0)
        return mu_hat


    def _get_reward_fn(self, W):
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
        return lambda s, a : W.T.dot(self._phi(s, a).T)


    def _optimize(self, mu_list):
        """linearly parametrize reward function.

        implements eq 11 from Abbeel
        note: we can rewrite this as an SVM problem

        Parameters
        ----------
        W : weight

        Returns
        -------
        TODO
        - think whether to do s, a or just s

        """
        logging.info("solving for W given mu_list")
        # define variables
        W = cvx.Variable(self._p)
        t = cvx.Variable(1)

        if self._use_slack:
            #xi = cvx.Variable(len(mu_list))
            xi = cvx.Variable(1)

        mu_exp = cvx.Parameter(self._p)
        mu_exp.value = self._mu_exp.flatten()

        if self._use_slack:
            C = cvx.Parameter(1)
            C = self._slack_penalty
            # since obj is max
            # we should penalize xi with minus
            # bc. xi bumps up expert's reward (see below)
            obj = cvx.Maximize(t - C * xi)
        else:
            obj = cvx.Maximize(t)

        constraints = []

        for mu in mu_list:
            if self._use_slack:
                # xi helps expert to perform better
                constraints += [W.T * mu_exp + xi >= W.T * mu + t]
            else:
                constraints += [W.T * mu_exp >= W.T * mu + t]
        constraints += [cvx.norm(W, 2) <= 1]

        prob = cvx.Problem(obj, constraints)
        prob.solve()

        # if svm formulation, need to normalize
        # W = W.value / np.linalg(W.value, 2)

        margin_v = t.value
        converged = margin_v <= self._precision

        # convergence in mu implies convergence in value (induced convergence)
        # but we don't use this relation here
        margin_mu = np.min(np.abs(np.array(mu_exp.value) - np.array(mu_list)))

        return np.array(W.value), (margin_v, margin_mu, converged)



