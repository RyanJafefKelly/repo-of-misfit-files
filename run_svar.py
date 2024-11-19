import os
import pickle as pkl

import flowjax.bijections as bij
import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpyro.distributions as dist
import numpyro.handlers as handlers
from flowjax.bijections import RationalQuadraticSpline  # type: ignore
from flowjax.distributions import (Normal, StandardNormal,  # type: ignore
                                   Uniform)
from flowjax.flows import coupling_flow  # type: ignore
from flowjax.train.data_fit import fit_to_data  # type: ignore
from jax.scipy.special import expit, logit
from numpyro.infer import MCMC, NUTS

from npe_convergence.examples.svar import (compute_summaries,  # TODO!
                                           run_inference, svar)
from npe_convergence.metrics import (kullback_leibler, total_variation,
                                     unbiased_mmd)


def run_svar(n_obs: int = 100, n_sims: int = 1_000):
    dirname = "res/svar/npe_n_obs_" + str(n_obs) + "_n_sims_" + str(n_sims) + "/"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    key = random.PRNGKey(1)
    true_params = jnp.array([0.579, -0.143, 0.836, 0.745, -0.660, -0.254, 0.1])
    # true_params = jnp.atleast_2d(true_params)
    y_obs = svar(key, true_params, n_obs=n_obs)
    y_obs_original = y_obs.copy()
    for i in range(6):
        plt.plot(y_obs[:, i])
        plt.savefig(f"svar_test_{i}.pdf")
        plt.clf()
    y_obs = compute_summaries(y_obs)
    key, sub_key = random.split(key)
    true_samples = run_inference(y_obs, n_obs, sub_key)
    true_thetas = true_samples['theta']
    # true_sigma = true_samples['sigma']  # TODO! NEED TO USE?
    # plot marginal posteriors
    # for i in range(6):
    #     plt.hist(true_thetas[:, i], bins=50)
    #     plt.savefig(f"true_samples_{i}.pdf")
    #     plt.clf()

    # TODO: sample prior
    key, sub_key = random.split(key)
    thetas = dist.Uniform(-.9, .9).sample(sub_key, (n_sims, 6))
    sigma = dist.Uniform(0, 1).sample(sub_key, (n_sims, 1))
    thetas_unbounded = logit((thetas + .9 ) / 1.8)
    sigma_unbounded = logit(sigma)
    thetas = jnp.hstack([thetas, sigma])
    thetas_unbounded = jnp.hstack([thetas_unbounded, sigma_unbounded])

    key, sub_key = random.split(key)
    # TODO: LAZY, annoying batch, TODO! CLEAN
    x_sims = jnp.empty((n_sims, 7))
    for i in range(n_sims):
        print(i)
        y = svar(sub_key, thetas[i, :], n_obs=n_obs)
        y = compute_summaries(jnp.atleast_2d(y))
        x_sims = x_sims.at[i, :].set(y)
    # svar_vmap = jax.vmap(svar, in_axes=(None, 0))
    # x_sims = svar_vmap(sub_key, thetas)
    # x_sims = jax.vmap(compute_summaries)(x_sims)
    # thetas = jnp
    # transform using logit to unbounded space
    # thetas_unbounded = thetas_unbounded
    thetas_mean = thetas_unbounded.mean(axis=0)
    thetas_std = thetas_unbounded.std(axis=0)
    thetas = (thetas_unbounded - thetas_mean) / thetas_std

    # sim_summ_data = x_sims.T
    sim_summ_data = x_sims  # TODO
    sim_summ_data_mean = sim_summ_data.mean(axis=0)
    sim_summ_data_std = sim_summ_data.std(axis=0)
    sim_summ_data = (sim_summ_data - sim_summ_data_mean) / sim_summ_data_std
    y_obs = (y_obs - sim_summ_data_mean) / sim_summ_data_std


    # TODO: train NN
    key, sub_key = random.split(sub_key)
    theta_dims = 7
    summary_dims = 7
    flow = coupling_flow(
        key=sub_key,
        base_dist=Normal(jnp.zeros(theta_dims)),
        # base_dist=Uniform(minval=-3 * jnp.ones(theta_dims), maxval=3 * jnp.ones(theta_dims)),
        transformer=RationalQuadraticSpline(knots=10, interval=5),  # 8 spline segments over [-3, 3].
        cond_dim=summary_dims,
        # flow_layers=8,  # NOTE: changed from 5, default is 8
        # nn_width=50,  # TODO: could experiment with
        # nn_depth=3  # TODO: could experiment with
        )
    key, sub_key = random.split(key)

    flow, losses = fit_to_data(
        key=sub_key,
        dist=flow,
        x=thetas,
        condition=sim_summ_data,
        learning_rate=5e-5,
        max_epochs=2000,
        max_patience=20,
        batch_size=256
    )

    plt.plot(losses['train'], label='train')
    plt.plot(losses['val'], label='val')
    plt.savefig(f'{dirname}losses.pdf')
    plt.clf()

    key, sub_key = random.split(key)

    num_posterior_samples = 10_000
    posterior_samples = flow.sample(sub_key, sample_shape=(num_posterior_samples,), condition=y_obs)
    posterior_samples = (posterior_samples * thetas_std) + thetas_mean
    posterior_samples = posterior_samples.at[:, :6].set( 1.8 * expit(posterior_samples[:, :6]) - 0.9)
    posterior_samples = posterior_samples.at[:, -1].set(expit(posterior_samples[:, -1]))
    # plt.xlim(0, 1)
    true_thetas = true_thetas.T  # TODO: ugly
    for i in range(7):
        _, bins, _ = plt.hist(true_thetas[:, i], bins=50, alpha=0.8, label='true')
        plt.hist(posterior_samples[:, i], bins=bins, alpha=0.8, label='NPE')
        plt.legend()
        plt.axvline(true_params[i], color='black')
        plt.savefig(f'{dirname}posterior_samples_{i}.pdf')
        plt.clf()

    # TODO: get samples
    return None

if __name__ == "__main__":
    run_svar()