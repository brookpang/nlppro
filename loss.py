# -*- coding:utf-8 -*-

import torch.nn as nn

class MultivariateNormalDiag():
    def __init__(self, loc=None, scale=None):
        self.loc = loc
        self.scale = scale

    def log_prob(self, z):
        normalization_constant = (-self.scale.log() - 0.5 * np.log(2 * np.pi))
        square_term = -0.5 * ((z - self.loc) / self.scale)**2
        log_prob_vec = normalization_constant + square_term
        return log_prob_vec.sum(1)

    def sample(self):
        # torch.randn_like
        z = self.loc + self.scale * torch.randn_like(self.scale)
        return z


def bayes_loss(output, labelemb):
    qz_x = MultivariateNormalDiag(output[:, :d],
                                  nn.functional.softplus(output[:, d:]))
    z = qz_x.sample()
    pz = MultivariateNormalDiag(torch.zeros_like(z), torch.ones_like(z))
    px_z = BernoulliVector(labelemb)
    return (px_z.log_prob(output) + pz.log_prob(z) - qz_x.log_prob(z)).mean()
