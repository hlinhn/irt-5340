#!/usr/bin/env python3

import torch


def standard_normal_log_pdf(x):
    mu = torch.zeros_like(x)
    scale = torch.ones_like(x)
    return torch.distributions.normal.Normal(mu, scale).log_prob(x)


def normal_log_pdf(x, mu, logvar):
    scale = torch.exp(0.5 * logvar)
    return torch.distributions.normal.Normal(mu, scale).log_prob(x)


def reparameterize_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mean)


def masked_bernoulli_log_pdf(x, mask, probs):
    dist = torch.distributions.bernoulli.Bernoulli(probs=probs)
    log_prob = dist.log_prob(x)
    return log_prob * mask.float()


def masked_gaussian_log_pdf(x, mask, mu, logvar):
    sigma = torch.exp(0.5 * logvar)
    dist = torch.distributions.normal.Normal(mu, sigma)
    log_prob = dist.log_prob(x)
    return log_prob * mask.float()


def kl_divergence_standard_normal_prior(z_mu, z_logvar):
    kl_div = -0.5 * (1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    kl_div = torch.sum(kl_div, dim=1)
    return kl_div
