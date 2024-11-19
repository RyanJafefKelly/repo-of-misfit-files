"""Run MA of order k model."""

import argparse
import os
import pickle as pkl

import arviz as az
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpyro  # type: ignore
import numpyro.distributions as dist  # type: ignore
from flowjax.bijections import RationalQuadraticSpline  # type: ignore
from flowjax.distributions import Normal  # type: ignore
from flowjax.flows import coupling_flow  # type: ignore
from flowjax.train.data_fit import fit_to_data  # type: ignore
from jax.scipy.special import expit, logit
from numpyro.infer import ESS, MCMC  # type: ignore

from npe_convergence.examples.mak import (MAK, generate_valid_samples,
                                          get_summaries,
                                          get_summaries_batches,
                                          numpyro_model)
from npe_convergence.metrics import (kullback_leibler, median_heuristic,
                                     unbiased_mmd)


def run_mak(*args, **kwargs):
    try:
        seed, n_obs, n_sims = args
        ma_order = kwargs['ma_order']
    except ValueError:
        args = args[0]
        seed = args.seed
        n_obs = args.n_obs
        n_sims = args.n_sims
        ma_order = args.ma_order
    dirname = "res/ma" + str(ma_order) + "/npe_n_obs_" + str(n_obs) + "_n_sims_" + str(n_sims) + "_seed_" + str(seed) + "/"
    print(f"Running MA of order {ma_order} model with seed: {seed}, n_obs: {n_obs}, n_sims: {n_sims}")
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # TODO: CHECK NORMALITY OF SAMPLE VARIANCE

    key = random.PRNGKey(seed)
    true_params = generate_valid_samples(key, ma_order, num_samples=1)
    true_params = true_params.ravel()
    # true_params = true_params[::-1]  # NOTE: reverse the order
    print("true_params: ", true_params)
    key, sub_key = random.split(key)
    x_obs = MAK(sub_key, true_params, n_obs=n_obs)
    x_obs = get_summaries(x_obs, ma_order)

    x_obs_original = x_obs.copy()

    num_posterior_samples = 10_000
    num_warmup = 10_000
    # nuts_kernel = NUTS(numpyro_model)
    ess_kernel = ESS(numpyro_model)
    thinning = 10
    num_chains = 2 * ma_order
    # num_chains = 4
    mcmc = MCMC(ess_kernel,
                num_warmup=num_warmup,
                num_samples=num_posterior_samples * thinning // num_chains,
                thinning=thinning,
                num_chains=num_chains,
                chain_method='vectorized'
                )
    init_params = jnp.tile(logit((true_params + 1) / 2), num_chains).reshape(num_chains, -1)
    key, sub_key = random.split(key)
    init_params = init_params + random.normal(sub_key, init_params.shape) * 0.1
    init_params = {'thetas': init_params}
    mcmc.run(random.PRNGKey(1),
             x_obs_original,
             init_params=init_params,
             n_obs=n_obs)
    mcmc.print_summary()
    true_posterior_samples = mcmc.get_samples()
    inference_data = az.from_numpyro(mcmc)

    az.plot_trace(inference_data, compact=False)
    plt.savefig(f"{dirname}traceplots.png")
    plt.close()
    az.plot_ess(inference_data, kind="evolution")
    plt.savefig(f"{dirname}ess_plots.png")
    plt.close()
    az.plot_autocorr(inference_data)
    plt.savefig(f"{dirname}autocorr.png")
    plt.close()

    # TODO: flow training
    # NOTE: transform ... over [-1, 1], but only train on valid samples
    # thetas = jnp.empty((n_sims, ma_order))
    # thetas_bounded = generate_valid_samples(ma_order, num_samples=n_sims)
    key, sub_key = random.split(key)
    thetas_bounded = random.uniform(sub_key, (n_sims, ma_order), minval=-1, maxval=1)
    thetas = logit((thetas_bounded + 1) / 2)

    key, sub_key = random.split(key)
    batch_size = min(1000, n_sims)
    sim_summ_data = get_summaries_batches(key, thetas_bounded, n_obs, n_sims, batch_size)

    thetas_mean = thetas.mean(axis=0)
    thetas_std = thetas.std(axis=0)
    thetas = (thetas - thetas_mean) / thetas_std

    sim_summ_data_mean = sim_summ_data.mean(axis=0)
    sim_summ_data_std = sim_summ_data.std(axis=0)
    sim_summ_data = (sim_summ_data - sim_summ_data_mean) / sim_summ_data_std

    x_obs = (x_obs - sim_summ_data_mean) / sim_summ_data_std

    key, sub_key = random.split(key)
    theta_dims = ma_order
    summary_dims = ma_order

    flow = coupling_flow(
        key=sub_key,
        base_dist=Normal(jnp.zeros(theta_dims)),
        # base_dist=Uniform(minval=-3 * jnp.ones(theta_dims), maxval=3 * jnp.ones(theta_dims)),
        transformer=RationalQuadraticSpline(knots=10, interval=5),  # 10 spline segments over [-5, 5].
        cond_dim=summary_dims,
        # flow_layers=8,  # NOTE: changed from 5, default is 8
        # nn_width=50,  # TODO: could experiment with
        nn_depth=2  # TODO: could experiment with
        )
    key, sub_key = random.split(key)

    flow, losses = fit_to_data(
        key=sub_key,
        dist=flow,
        x=thetas,
        condition=sim_summ_data,
        learning_rate=5e-4,  # TODO: could experiment with
        max_epochs=500,
        max_patience=10,
        batch_size=256,
    )

    plt.plot(losses['train'], label='train')
    plt.plot(losses['val'], label='val')
    plt.savefig(f'{dirname}losses.pdf')
    plt.clf()

    key, sub_key = random.split(key)

    posterior_samples_original = flow.sample(sub_key, sample_shape=(num_posterior_samples,), condition=x_obs)
    posterior_samples = (posterior_samples_original * thetas_std) + thetas_mean

    posterior_samples = expit(posterior_samples) * 2 - 1

    posterior_samples = jnp.squeeze(posterior_samples)
    true_posterior_samples = true_posterior_samples['thetas']
    for i in range(ma_order):
        _, bins, _ = plt.hist(posterior_samples[:, i], bins=50)
        plt.hist(true_posterior_samples[:, i], bins=bins, alpha=0.5)
        plt.axvline(true_params[i], color='red')
        plt.savefig(f'{dirname}hist_{i}.pdf')
        plt.clf()

    kl = kullback_leibler(true_posterior_samples, posterior_samples)

    lengthscale = median_heuristic(jnp.vstack([true_posterior_samples,
                                               posterior_samples]))

    mmd = unbiased_mmd(true_posterior_samples, posterior_samples, lengthscale)

    with open(f'{dirname}posterior_samples.pkl', 'wb') as f:
        pkl.dump(posterior_samples, f)

    with open(f'{dirname}true_posterior_samples.pkl', 'wb') as f:
        pkl.dump(true_posterior_samples, f)

    with open(f'{dirname}kl.txt', 'w') as f:
        f.write(str(kl))

    with open(f'{dirname}mmd.txt', 'w') as f:
        f.write(str(mmd))

    # TODO: coverage
    num_coverage_samples = 100
    coverage_levels = [0.8, 0.9, 0.95]

    # bias/coverage for true parameter
    true_params_unbounded = logit((true_params + 1) / 2)
    true_params_standardised = (true_params_unbounded - thetas_mean) / thetas_std
    bias = jnp.mean(posterior_samples, axis=0) - true_params
    pdf_posterior_samples = flow.log_prob(posterior_samples_original,
                                          x_obs)
    pdf_posterior_samples = jnp.sort(pdf_posterior_samples.ravel(),
                                     descending=True)
    pdf_theta = flow.log_prob(true_params_standardised, x_obs)
    true_in_credible_interval = [0, 0, 0]
    for i, level in enumerate(coverage_levels):
        coverage_index = int(level * num_posterior_samples)
        pdf_posterior_sample = pdf_posterior_samples[coverage_index]
        if pdf_theta > pdf_posterior_sample:
            true_in_credible_interval[i] = 1

    with open(f"{dirname}true_in_credible_interval.txt", "w") as f:
        f.write(f"{true_in_credible_interval}\n")

    with open(f"{dirname}true_bias.txt", "w") as f:
        f.write(f"{bias}\n")

    coverage_levels_counts = [0, 0, 0]
    biases = jnp.array([])

    for i in range(num_coverage_samples):
        key, sub_key = random.split(key)
        theta_draw_original = dist.Uniform(-1, 1).sample(sub_key, (1, ma_order))

        theta_draw = logit((theta_draw_original+1) / 2)
        theta_draw = (theta_draw - thetas_mean) / thetas_std

        key, sub_key = random.split(key)
        x_draw = get_summaries_batches(sub_key, theta_draw_original,
                                       n_obs=n_obs, n_sims=1, batch_size=1)
        # x_draw = jnp.squeeze(x_draw)

        x_draw = (x_draw - sim_summ_data_mean) / sim_summ_data_std

        posterior_samples_original = flow.sample(sub_key,
                                                 sample_shape=(num_posterior_samples,),
                                                 condition=x_draw)
        posterior_samples = (posterior_samples_original * thetas_std) + thetas_mean
        posterior_samples = jnp.squeeze(posterior_samples)
        posterior_samples = expit(2*posterior_samples - 1)

        bias = jnp.mean(posterior_samples, axis=0) - theta_draw_original
        biases = jnp.concatenate((biases, bias.ravel()))
        pdf_posterior_samples = flow.log_prob(posterior_samples_original,
                                              x_draw)
        pdf_posterior_samples = jnp.sort(pdf_posterior_samples.ravel(),
                                         descending=True)
        pdf_theta = flow.log_prob(theta_draw, x_draw)

        for i, level in enumerate(coverage_levels):
            coverage_index = int(level * num_posterior_samples)
            pdf_posterior_sample = pdf_posterior_samples[coverage_index]
            if pdf_theta > pdf_posterior_sample:
                coverage_levels_counts[i] += 1

    print(coverage_levels_counts)
    estimated_coverage = jnp.array(coverage_levels_counts)/num_coverage_samples

    with open(f"{dirname}coverage.txt", "w") as f:
        f.write(f"{estimated_coverage}\n")

    with open(f"{dirname}biases.txt", "w") as f:
        f.write(f"{biases}\n")

    return kl, mmd


if __name__ == "__main__":
    numpyro.set_host_device_count(4)
    parser = argparse.ArgumentParser(
        prog="run_mak.py",
        description="Run MA of order k model.",
        epilog="Example usage: python run_mak.py"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_obs", type=int, default=500)
    parser.add_argument("--n_sims", type=int, default=30_000)
    parser.add_argument("--ma_order", type=int, default=6)
    args = parser.parse_args()
    run_mak(args)
