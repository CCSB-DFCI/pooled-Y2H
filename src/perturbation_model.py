import json
import math

import numpy as np
from scipy import stats, optimize
from matplotlib import pyplot as plt
import seaborn as sns

from data_processing import load_combined_pooled_Y2H_dataset


def log_likelihood_unperturbed(log2fc, error_log2fc, non_perturbation_sigma):
    # in the scipy API: loc is mean, scale is stddev
    loglikelihood_unperturbed_model = stats.norm.logpdf(
        log2fc,
        loc=0,
        scale=(error_log2fc**2
              + non_perturbation_sigma**2
              ) ** 0.5,
    )
    return loglikelihood_unperturbed_model


def log_likelihood_perturbed(log2fc, error_log2fc, perturbation_size, perturbation_sigma):
    ASSUMED_PERTURBATION_BOUND = -1

    var_prior = perturbation_sigma**2
    var_measurement = error_log2fc**2
    sigma_combined = np.sqrt(var_prior + var_measurement)

    # Marginal (untruncated) distribution of y
    log_py = stats.norm.logpdf(
        log2fc, loc=perturbation_size, scale=sigma_combined
    )

    # Posterior of theta | y under untruncated normal prior
    var_post = 1.0 / (1.0 / var_prior + 1.0 / var_measurement)
    sigma_post = np.sqrt(var_post)
    mu_post = var_post * (
        perturbation_size / var_prior + log2fc / var_measurement
    )
    t = (ASSUMED_PERTURBATION_BOUND - mu_post) / sigma_post
    # Probability that θ ≤ bound under this posterior
    log_post_cdf = stats.norm.logcdf(t)

    # Prior truncation normalising constant
    alpha = (
        ASSUMED_PERTURBATION_BOUND - perturbation_size
    ) / perturbation_sigma
    log_Z = stats.norm.logcdf(alpha)

    loglikelihood_perturbed_model = log_py + log_post_cdf - log_Z
    return loglikelihood_perturbed_model


def nll_mixture_model(params, log2fc, error_log2fc):

    (perturbation_size, 
     perturbation_sigma, 
     non_perturbation_sigma,
     relative_composition) = params

    # in the scipy API: loc is mean, scale is stddev
    loglikelihood_unperturbed_model = log_likelihood_unperturbed(
        log2fc,
        error_log2fc,
        non_perturbation_sigma
    )

    loglikelihood_perturbed_model = log_likelihood_perturbed(
        log2fc,
        error_log2fc,
        perturbation_size,
        perturbation_sigma
    )

    nll = -np.sum(np.logaddexp(
        np.log(relative_composition) + loglikelihood_perturbed_model,
        np.log(1 - relative_composition) + loglikelihood_unperturbed_model
    ))

    return nll


def perturbation_LLR(log2fc, error_log2fc, params):
    (perturbation_size, 
     perturbation_sigma, 
     non_perturbation_sigma,
     relative_composition) = params
    loglikelihood_unperturbed_model = log_likelihood_unperturbed(
        log2fc,
        error_log2fc,
        non_perturbation_sigma
    )
    loglikelihood_perturbed_model = log_likelihood_perturbed(
        log2fc,
        error_log2fc,
        perturbation_size,
        perturbation_sigma
    )
    return (loglikelihood_unperturbed_model - loglikelihood_perturbed_model)


def fit_perturbation_model(out_path="../output/fitted_mixture_model_params.json"):
    print("Fitting perturbation model to combined dataset")
    
    df = load_combined_pooled_Y2H_dataset(drop_superceded=True,
                                          add_alphafold=False,
                                          add_perturbation_llr=False,  # important otherwise will be infinite loop
                                          )
    log2fc = df["log2FC_combined"].dropna().to_numpy()
    error_log2fc = df["error_log2FC_combined"].dropna().to_numpy()

    initial_guess = [
        -5.0, # perturbation_size
        1.5,  # perturbation_sigma
        0.5,  # non_perturbation_sigma
        0.3]  # relative_composition
    bounds = [
        (-10.0, -2),   # perturbation_size
        (0.1, 10.0),   # perturbation_sigma
        (0.1, 10.0),   # non_perturbation_sigma
        (0.05, 0.95)   # relative_composition
    ]
    res = optimize.minimize(nll_mixture_model, 
                            initial_guess, 
                            bounds=bounds, 
                            method='L-BFGS-B', 
                            args=(log2fc, error_log2fc))
    print(res)

    (fit_perturbation_size, 
    fit_perturbation_distribution, 
    fit_non_perturbation_distribution, 
    fit_relative_composition) = res.x

    plot_fitted_mixture_model(res, log2fc, error_log2fc)

    # NOTE: keep the order the same as in the optimized function above
    fit_params = {
        'perturbation_mu': fit_perturbation_size, 
        'perturbation_sigma': fit_perturbation_distribution, 
        'non_perturbation_sigma': fit_non_perturbation_distribution, 
        'proportion_of_perturbation': fit_relative_composition
    }
    with open(out_path, 'w') as f:
        json.dump(fit_params, f, indent=2)


def plot_fitted_mixture_model(res, log2fc, error_log2fc):
    (fit_perturbation_size, 
    fit_perturbation_distribution, 
    fit_non_perturbation_distribution, 
    fit_relative_composition) = res.x 
    fig, ax = plt.subplots(figsize=(8,6))
    sns.histplot(log2fc, 
                bins=100, 
                stat='density', 
                ax=ax, 
                color='lightgrey', 
                edgecolor='black')
    xmin = math.floor(min(log2fc))
    xmax = math.ceil(max(log2fc))
    ax.set_xlim(xmin, xmax)
    for pos in ['right', 'top']:
        ax.spines[pos].set_visible(False)
    fig.savefig('../output/figures/Log2FC-distribution_hist.pdf',
                bbox_inches='tight')

    x_plot = np.linspace(xmin, xmax, 1000)
    y_wt = (1 - fit_relative_composition) * np.exp(
        log_likelihood_unperturbed(
            x_plot,
            error_log2fc.mean(),
            fit_non_perturbation_distribution
        )
    )
    y_ptrb = fit_relative_composition * np.exp(
        log_likelihood_perturbed(
            x_plot,
            error_log2fc.mean(),
            fit_perturbation_size,
            fit_perturbation_distribution
        )
    )
    y_total = y_wt + y_ptrb

    llr = perturbation_LLR(
        x_plot,
        error_log2fc.mean(),
        res.x
    )
    cutoff_wt = np.interp(2, llr, x_plot)
    cutoff_prtb = np.interp(-2, llr, x_plot)

    ax.plot(x_plot, y_wt, color='#1D7AAB', lw=2, label='Fitted WT')
    ax.plot(x_plot, y_ptrb, color='#CA7682', lw=2, label='Fitted Mut')
    ax.plot(x_plot, y_total, color='black', linestyle='--', lw=1, label='Total Mixture')
    ax.set_xlim(xmin, xmax)
    fig.savefig('../output/figures/Log2FC-distribution_fitted-mixture-model.pdf',
                bbox_inches='tight')

    ax.axvspan(xmin, cutoff_prtb, color='#CA7682', alpha=0.3, linewidth=0)
    ax.axvspan(cutoff_wt, xmax, color='#1D7AAB', alpha=0.3, linewidth=0)
    ax.axvspan(cutoff_prtb, cutoff_wt, color='grey', alpha=0.3, linewidth=0)
    llr_vals = perturbation_LLR(
        log2fc,
        error_log2fc,
        res.x
    )
    ax.text(x=(cutoff_prtb - cutoff_wt) / 2 + cutoff_wt,
            y=ax.get_ylim()[1] * 0.8,
            s=f'{((llr_vals > -2) & (llr_vals < 2)).mean():.0%} Uncertain',
            ha='center',
            va='center',
            fontsize=16,
            rotation=90,
            fontweight='bold',
            fontdict={'color': 'white'})
    ax.text(x=(xmin - cutoff_prtb) / 2 + cutoff_prtb,
            y=ax.get_ylim()[1] * 0.8,
            s=f'{(llr_vals < -2).mean():.0%} Perturbed',
            ha='center',
            va='center',
            fontsize=16,
            fontweight='bold',
            fontdict={'color': 'white'})
    ax.text(x=(xmax - cutoff_wt) / 2 + cutoff_wt,
            y=ax.get_ylim()[1] * 0.8,
            s=f'{(llr_vals > 2).mean():.0%} Unperturbed',
            ha='center',
            va='center',
            fontsize=16,
            fontweight='bold',
            fontdict={'color': 'white'})
    fig.savefig('../output/figures/Log2FC-distribution_fitted-mixture-model_with_average_cutoffs.pdf',
                bbox_inches='tight')
