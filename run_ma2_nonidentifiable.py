import os

import flowjax.bijections as bij
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
from flowjax.bijections import RationalQuadraticSpline  # type: ignore
from flowjax.distributions import (Normal, StandardNormal,  # type: ignore
                                   Uniform)
from flowjax.flows import CouplingFlow  # type: ignore
from flowjax.train.data_fit import fit_to_data  # type: ignore
from jax.scipy.special import expit, logit

from npe_convergence.examples.ma2 import (MA2, CustomPrior_t1, CustomPrior_t2,
                                          autocov)


def run_ma2():
    # TODO: num sim, n_obs, effects
    # TODO! TRANSFORM YOUR VARIABLES!!
    print('t')
    key = random.PRNGKey(0)
    num_sims = 10_000
    true_params = jnp.array([1.5, -0.5])
    n_obs = 100
    y_obs = MA2(*true_params, n_obs=n_obs, key=key)
    y_obs = jnp.array([[jnp.var(y_obs)], autocov(y_obs, lag=1), autocov(y_obs, lag=2)]).ravel()
    y_obs_original = y_obs.copy()

    # test_t1 = CustomPrior_t1.rvs(2., size=(1,), random_state=10)
    # test_t2 = CustomPrior_t2.rvs(test_t1, 1., size=(1,), random_state=10)

    key, sub_key = random.split(key)
    t1_bounded = 4 * random.uniform(sub_key, shape=(num_sims,)) - 2
    t1 = logit((t1_bounded + 2) / 4)

    key, sub_key = random.split(key)
    t2_bounded = 2 * random.uniform(sub_key, shape=(num_sims,)) - 1
    t2 = logit((t2_bounded + 1) / 2)

    key, sub_key = random.split(key)
    sim_data = MA2(t1, t2, n_obs=n_obs, batch_size=num_sims, key=sub_key)
    # sim_summ_data = sim_data
    sim_summ_data = jnp.array((jnp.var(sim_data, axis=1), autocov(sim_data, lag=1), autocov(sim_data, lag=2)))

    thetas = jnp.column_stack([t1, t2])
    # thetas = jnp.vstack([thetas, true_params])
    thetas_mean = thetas.mean(axis=0)
    thetas_std = thetas.std(axis=0)
    thetas = (thetas - thetas_mean) / thetas_std

    sim_summ_data = sim_summ_data.T
    # sim_summ_data = jnp.vstack([sim_summ_data, y_obs])
    sim_summ_data_mean = sim_summ_data.mean(axis=0)
    sim_summ_data_std = sim_summ_data.std(axis=0)
    sim_summ_data = (sim_summ_data - sim_summ_data_mean) / sim_summ_data_std

    y_obs = (y_obs - sim_summ_data_mean) / sim_summ_data_std

    key, sub_key = random.split(sub_key)
    theta_dims = 2
    summary_dims = 3
    flow = CouplingFlow(
        key=sub_key,
        base_dist=Normal(jnp.zeros(theta_dims)),
        # base_dist=Uniform(minval=-3 * jnp.ones(theta_dims), maxval=3 * jnp.ones(theta_dims)),
        transformer=RationalQuadraticSpline(knots=10, interval=5),  # 8 spline segments over [-3, 3].
        cond_dim=summary_dims,
        # flow_layers=8,  # NOTE: changed from 5, default is 8
        # nn_width=50,  # TODO: could experiment with
        # nn_depth=3  # TODO: could experiment with
        )
#     flow = masked_autoregressive_flow(
#         subkey,
#         base_dist=Normal(jnp.zeros(theta_dims)),
#         transformer=bij.Affine(),
#         cond_dim=summary_dims,
# )


    key, sub_key = random.split(key)

    flow, losses = fit_to_data(
        key=sub_key,
        dist=flow,
        x=thetas,
        condition=sim_summ_data,
        learning_rate=5e-4,
        max_epochs=500,
        max_patience=20,
        batch_size=256
    )

    plt.plot(losses['train'], label='train')
    plt.plot(losses['val'], label='val')
    plt.savefig('losses.pdf')
    plt.clf()

    key, sub_key = random.split(key)

    num_ppc_samples = 2_000
    posterior_samples = flow.sample(sub_key, sample_shape=(num_ppc_samples,), condition=y_obs)
    posterior_samples = (posterior_samples * thetas_std) + thetas_mean
    posterior_samples.at[:, 0].set(4 * expit(posterior_samples[:, 0]) - 2)
    posterior_samples.at[:, 1].set(2 * expit(posterior_samples[:, 1]) - 1)
    plt.hist(posterior_samples[:, 0], bins=50)
    # plt.xlim(0, 1)
    plt.axvline(true_params[0], color='red')
    plt.savefig('t1_posterior_nonidentifiable.pdf')
    plt.clf()

    plt.hist(posterior_samples[:, 1], bins=50)
    plt.axvline(true_params[1], color='red')
    # plt.xlim(0, 1)
    plt.savefig('t2_posterior_nonidentifiable.pdf')
    plt.clf()

    resolution = 200
    xgrid, ygrid = jnp.meshgrid(jnp.linspace(-3, 3, resolution), jnp.linspace(-3, 3, resolution))
    xy_input = jnp.column_stack([xgrid.ravel(), ygrid.ravel()])
    zgrid = jnp.exp(flow.log_prob(xy_input, condition=y_obs)).reshape(resolution, resolution)
    # restandardise = lambda x: x * thetas_std + thetas_mean
    xgrid = xgrid.ravel() * thetas_std[0] + thetas_mean[0]
    xgrid = 4 * expit(xgrid) - 2
    xgrid = xgrid.reshape(resolution, resolution)
    ygrid = ygrid.ravel() * thetas_std[1] + thetas_mean[1]
    ygrid = 2 * expit(ygrid) - 1
    ygrid = ygrid.reshape(resolution, resolution)
    plt.axvline(true_params[0], color='red')
    plt.axhline(true_params[1], color='red')
    plt.contourf(xgrid, ygrid, zgrid, levels=50)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig('contour_nonidentifiable.pdf')
    plt.clf()

    num_ppc_samples = 2_000
    ppc_samples = MA2(posterior_samples[:, 0], posterior_samples[:, 1], batch_size=num_ppc_samples, n_obs=n_obs, key=sub_key)
    ppc_summaries = jnp.array((autocov(ppc_samples, lag=1), autocov(ppc_samples, lag=2)))
    plt.hist(ppc_summaries[0], bins=50)
    plt.axvline(y_obs_original[0], color='red')
    plt.savefig('ppc_var_nonidentifiable.pdf')
    plt.clf()

    plt.hist(ppc_summaries[-2], bins=50)
    plt.axvline(y_obs_original[-2], color='red')
    plt.savefig('ppc_ac1_nonidentifiable.pdf')
    plt.clf()

    plt.hist(ppc_summaries[-1], bins=50)
    plt.axvline(y_obs_original[-1], color='red')
    plt.savefig('ppc_ac2_nonidentifiable.pdf')
    plt.clf()

    return


if __name__ == '__main__':
    print(os.getcwd())
    run_ma2()
