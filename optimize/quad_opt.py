import numpy as np
from sklearn.svm import LinearSVC, SVC


class QuadOpt():
    def __init__(self, epsilon=0.01, penalty=0, hyperplane_margin=False):
        self.mus = []
        self.epsilon = epsilon
        self.penalty = penalty
        self.hyperplane_margin = hyperplane_margin

    def transform_data(self, target_mu, cur_mu):
        X = np.array(self.mus + [target_mu])
        y = np.empty(len(X))
        y.fill(-1)
        # target mu is labeled +1
        y[-1] = 1
        return X, y

    def optimize(self, target_mu, cur_mu, normalize=True):
        self.mus.append(cur_mu)
        X,y = self.transform_data(target_mu, cur_mu)
        #clf = LinearSVC(C=self.penalty)
        clf = SVC(kernel='linear', C=self.penalty)
        clf.fit(X,y)
        # since decision hyperplane is W^T mu = 0
        # coefficients is a normal vector to hyperplane
        W = clf.coef_[0]
        # TODO: better handle the case of W = 0
        # than this hack
        if np.all(W == 0.0):
            converged = True
            margin = 0
            return W, converged, margin


        norm = np.linalg.norm(W, 2)
        if normalize:
            W = W / norm
        # taken from Abbeel (2004)
        # dist from a support vector to mu_expert
        diffs = target_mu - np.array(self.mus)
        # TODO: check if abs can be applied
        # otherwise margin can be negative
        if self.hyperplane_margin:
            margin = 1 / np.linalg.norm(W, 2)
        else:
            margin = np.abs(W.dot(diffs.T)).min()
        converged = margin < self.epsilon
        return W, converged, margin

