import observables as obs
import boltzmann_gen as bg
import numpy as np
import jax
import jax.numpy as jnp

def res_N_mcmc(trajectory, V, N_samples, tolerance, window, c, abs_tol):
    """Per catene correlate: MCMC standard e Flow-MCMC."""
    results = {"N": [], "E_mean": [], "E_mean_err": [], "tau_x": []}
    for N in N_samples:
        sub_traj = trajectory[:N]
        energies = np.array(jax.vmap(V)(sub_traj))
        E_mean_err, _, _ = obs.blocking_analysis(energies, window, tolerance, abs_tol, obs="E_mean")
        tau_x = obs.tau_int(np.array(sub_traj[:, 0]), c)
        results["N"].append(N)
        results["E_mean"].append(np.mean(energies))
        results["E_mean_err"].append(E_mean_err)
        results["tau_x"].append(tau_x)
    return results


def res_N_bg(x_bg, log_qx, V, T, kb, N_samples):
    """Per il Boltzmann Generator: richiede reweighting per ogni N."""
    results = {"N": [], "E_mean": [], "E_mean_err": [], "ESS_normalized": [], "tau_eff": []}
    for N in N_samples:
        x_sub = x_bg[:N]
        logq_sub = log_qx[:N]

        weights = bg.reweight_samples(x_sub, logq_sub, T=T, potential=V, kb=kb)
        E_values = jax.vmap(V)(x_sub)
        Ess = bg.Ess(weights)

        E_mean = jnp.sum(weights * E_values)
        var_E = jnp.sum(weights * (E_values - E_mean) ** 2)
        E_mean_err = jnp.sqrt(var_E / Ess)

        results["N"].append(N)
        results["E_mean"].append(float(E_mean))
        results["E_mean_err"].append(float(E_mean_err))
        results["ESS_normalized"].append(float(Ess / N))
        results["tau_eff"].append(float(N / (2.0 * Ess)))
    return results