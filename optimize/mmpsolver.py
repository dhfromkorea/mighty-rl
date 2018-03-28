import numpy as np
from cvxpy import *

from optimize.mmp_loss import action_variation_loss

class MMTOpt():
    def __init__(self, pi_expert,
    	               slack_scale,
                       alpha,
                       use_slack,
                       state_histogram=[],
                       w_norm=2,
                       loss_type='qvs',
                       slack_norm=1,
                       fixed_loss=1):
        '''
        Parameters
            loss_type(str): qvs (Q Variation Stochastic)
                            qvg (Q vairiation Greedy)
                            fixed (scalar loss) for testing
        todo: alpha for loss
        hyperparameters make available in optimize function
        '''
        self.pi_expert = pi_expert
        self.state_histogram = state_histogram
        self.w_norm = w_norm
        self.mu_irls = []
        self.pi_losses = []
        self.pi_irls = []
        self.fixed_loss = fixed_loss

        #Flags for structured and slack structured max margin.
        self.loss_type = loss_type
        self.use_slack = use_slack
        self.gamma = slack_scale
        self.q = slack_norm
        self.alpha = alpha

    def optimize(self, mu_expert,
                       mu_irl,
                       pi_irl):

        self.mu_irls.append(mu_irl)

        #Add the next element to loss.
        if self.loss_type == 'qvs':
            loss = action_variation_loss(Q_expert=self.pi_expert.Q,
                                         Q_irl=pi_irl.Q,
                                         state_histogram=self.state_histogram)
        elif self.loss_type == 'qvg':
            raise NotImplementedError()

        elif self.loss_type == 'fixed':
            loss = self.fixed_loss

        self.pi_losses.append(loss)

        return self.mmpsolver(mu_expert, self.mu_irls)

    def mmpsolver(self, mu_expert, mu_irls):
        """
            Maximum Margin Planning Algorithm
            solve a QP

            min w.r.t. W, Z
            (1/2) ||w||^2 + slack_scale * (s)^slack_norm
            s.t w.mu(expert) >= w.mu(candidate) + loss(candidate) - s forall candidates
        """
        num_features = mu_expert.shape[0]
        # optimization variables
        w = Variable(num_features)
        # hyperparameters
        alpha = Parameter(sign='Positive', value=self.alpha)
        gamma = Parameter(sign='Positive', value=self.gamma)

        if self.use_slack:
            s = Variable()
            # set slack positive
            constraints = [w.T * mu_expert >= (w.T * mu) + alpha* L - s for mu, L in zip(self.mu_irls, self.pi_losses)]
            objective = Minimize(0.5 * norm(w, self.w_norm) + gamma * norm(s, self.q))
            #constraints.append(s >= 0)
        else:
            constraints = [w.T * mu_expert >= (w.T * mu) + alpha*L for mu, L in zip(self.mu_irls, self.pi_losses)]
            objective = Minimize(0.5 * norm(w, self.w_norm))


        prob = Problem(objective, constraints)
        # todo: prob.status == infeasible
        # graceful exit
        prob.solve()

        #print('losses', np.array(self.pi_losses))
        #print('alphas losses', self.alpha * np.array(self.pi_losses))

        res = {}
        res['w'] = np.array(w.value).squeeze()
        diffs = res['w'].dot(np.vstack(mu_expert) - np.array(self.mu_irls).T)
        #print('diffs', diffs)
        # todo fix margin
        min_i = np.argmin(np.abs(diffs))

        res['margin'] = diffs[min_i]
        res['s'] = None
        if self.use_slack:
            res['s'] = s.value
            diffs = np.vstack(mu_expert) - np.array(self.mu_irls).T
            res['margin'] = res['margin'] - res['s'] + self.pi_losses[min_i]
        #print(self.pi_losses[min_i])
        return res 
