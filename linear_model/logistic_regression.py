import numpy as np
import optimise.gradient_descent as gd
from sklearn.preprocessing import add_dummy_feature

class LogisticRegression():
    def __init__(self, penalty='l2', reg_strength=1.0, fit_intercept=True, tol=1e-6, rate=1e-6, max_iter=10000):
        self.penalty = penalty # only L2 currently supported
        self.reg_strength = reg_strength
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.rate = rate
        self.max_iter = max_iter

    def gradient(self, beta, x, y):
        odds = np.exp(np.matmul(x, beta))
        prob = odds / (1 + odds)
        # TODO? Set prob to 1 if np.isinf(odds)
        penalty = 0
        if self.penalty is not None:
            penalty = self.reg_strength * beta
        return np.matmul(x.transpose(), prob - y) + penalty

    def fit(self, x, y, beta=None, keep_intermediate_betas=False, verbose=False):
        self.beta = beta
        if self.fit_intercept:
            x = add_dummy_feature(x)
        if self.beta is None:
            self.beta = np.zeros(x.shape[1])
        gd_ret = gd.gradient_descent(self.beta, self.gradient, x, y, rate=self.rate, tol=self.tol, verbose=verbose, ret_intermediate_betas=keep_intermediate_betas)
        self.betas = None
        self.beta = gd_ret
        if keep_intermediate_betas:
            self.betas = gd_ret
            self.beta = gd_ret[-1]
        return self

    @staticmethod
    def xbeta(x, beta):
        return np.matmul(x, beta)

    def predict(self, x, p_threshold=0.5):
        if self.fit_intercept and x.shape[1] < len(self.beta):
            x = add_dummy_feature(x)
        odds_threshold = p_threshold / (1.0 - p_threshold)
        return (np.exp(LogisticRegression.xbeta(x, self.beta)) > odds_threshold).astype(int)

    @staticmethod
    def _likelihood(x, y, beta):
        xb = LogisticRegression.xbeta(x, beta)
        return np.matmul(y.transpose(), xb) - np.log(1 + np.exp(xb)).sum()

    @staticmethod
    def _loss(x, y, beta, reg_strength):
        return -LogisticRegression._likelihood(x, y, beta) + reg_strength * 0.5 * np.dot(beta, beta)

    def likelihood(self, x, y, use_intermediate_betas=False):
        if hasattr(self, 'beta'):
            if self.fit_intercept and x.shape[1] < len(self.beta):
                x = add_dummy_feature(x)
            if use_intermediate_betas:
                if not(hasattr(self, 'betas')):
                    message = 'Cannot return likelihood(s) evaluated using unconverged betas if those were not retained. Please call fit() first with keep_intermediate_betas=True.'
                    raise ValueError(message)
                likelihoods = np.zeros(len(self.betas))
                for i in range(len(self.betas)):
                    # Nothing else works: self, self.__class__ (which should be equivalent), just plain _likelihood...
                    likelihoods[i] = LogisticRegression._likelihood(x, y, self.betas[i])
                return likelihoods
            return LogisticRegression._likelihood(x, y, self.beta)
        else:
            raise RuntimeError("This LogisticRegression instance isn't fitted yet")

    def loss(self, x, y, use_intermediate_betas=False):
        if hasattr(self, 'beta'):
            if self.fit_intercept and x.shape[1] < len(self.beta):
                x = add_dummy_feature(x)
            if use_intermediate_betas:
                if not(hasattr(self, 'betas')):
                    message = 'Cannot return loss(es) evaluated using unconverged betas if those were not retained. Please call fit() first with keep_intermediate_betas=True.'
                    raise ValueError(message)
                losses = np.zeros(len(self.betas))
                for i in range(len(self.betas)):
                    # Nothing else works: self, self.__class__ (which should be equivalent), just plain _likelihood...
                    losses[i] = LogisticRegression._loss(x, y, self.betas[i], self.reg_strength)
                return losses
            return LogisticRegression._loss(x, y, self.beta, self.reg_strength)
        else:
            raise RuntimeError("This LogisticRegression instance isn't fitted yet")

