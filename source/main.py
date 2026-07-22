import mcmc as metro
import boltzmann_gen as bg
import mcmc_boltzmann as fmcmc
import plot as plot
import observables as obs
import vsN as vsN

import optax
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

def main():
    # Simulation parameters
    T_max = 1100.0   # highest temperature
    T_min = 100.0    # lowest temperature
    T_step = 10.0     # temperature step
    Ts = jnp.arange(T_max, T_min-T_step, -T_step)  # from hot to cold
    T_analysis = [Ts[0], Ts[len(Ts)//2], Ts[-1]] # temperature for R(m) and N analysis 
    D = 3 # dimension of the system fixed to 3 for the double well potential: 1 slow variable + 2 fast variables
    kb = 8.617333262145e-5
    key = jax.random.PRNGKey(42)
    n_samples = 10**6 # these are the number of samples for the Boltzmann generator and the steps of MCMC
    n_train_steps = 1000 # number of training steps for the Boltzmann generator at each temperature
    n_samples_training = 500 # number of samples for the training of the Boltzmann generator at each temperature

    # potential parameters
    a = 0.1
    b = 1.0
    l = 0.0
    V = obs.make_potential(a, b, l, "simmetric_double_well")

    results_mcmc = {
        "T": [], "E_mean": [], "E_mean_err": [], "acceptance": [],
        "tau_x": [], "delta_tau": [], "mean_sign": [], "R_list": [],
        "ESS_normalized": []
    }
    results_bg = {
        "T": [], "E_mean": [], "E_mean_err": [], "tau_eff": [],
        "mean_sign": [], "loss_last": [], "loss_start": [], "ESS_normalized": []
    }
    results_mcmc_bg = {
        "T": [], "E_mean": [], "E_mean_err": [], "acceptance": [],
        "tau_x": [], "delta_tau": [], "mean_sign": [], "R_list": [],
        "ESS_normalized": []
    }

    # full-dataset for T in T_analysis
    traj_mcmc_full = {}
    pos_bg_full = {}
    logqx_bg_full = {}
    traj_mcmc_bg_full = {}
    losses_bg_full = {}

    ####### 0. INITIALIZATION #######
    tollerance = 0.1
    window = 3
    c = 5
    abs_tol = 0.01
    n_thermalization = 5000
    step_size = 0.1

    gen_config = metro.make_config_generator(D, "normal")
    run_first_therm = metro.make_simulation(D, n_thermalization*10, V, kb)
    run_low_therm = metro.make_simulation(D,n_thermalization*100, V, kb)
    run_therm  = metro.make_simulation(D, n_thermalization, V, kb)
    run_prod   = metro.make_simulation(D, n_samples, V, kb)
    run_flow_prod = fmcmc.make_flow_simulation_batched(n_samples, V, kb)

    ###### 1. MCMC SIMULATION ######
    x, key = gen_config(key)
    for T in tqdm(Ts, desc="MCMC", unit="T"):
        if T == Ts[0]:
            _, _, key, x = run_first_therm(key, T, step_size, x)
        if T <= 200:
            _, _, key, x = run_low_therm(key, T, step_size, x)
        _, _, key, x = run_therm(key, T, step_size, x)
        trajectory, acceptance_rate, key, x = run_prod(key, T, step_size, x)

        obs.append_observables(results_mcmc, T, trajectory, acceptance_rate, V,
                                tollerance, window, c, abs_tol, kb)

        if float(T) in T_analysis:
            traj_mcmc_full[float(T)] = trajectory  

    print("MCMC simulation completed.")

    ###### 2. BOLTZMANN GENERATOR ######
    key, subkey = jax.random.split(key)
    bg_params, bg_static = bg.init_params(subkey, dim=D, n_layers=6, hidden_dim=32)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(bg_params)
    train_T = bg.make_train_loop(optimizer, potential=V, kb=kb)

    for T in tqdm(Ts, desc="BG and MCMC-BG", unit="T"):
        key, subkey = jax.random.split(key)
        bg_params, opt_state, key, losses = train_T(
            bg_params, bg_static, opt_state, subkey, T,
            n_samples=n_samples_training, n_steps=n_train_steps)

        x_bg, key = bg.sample(bg_params, bg_static, key, n_samples=n_samples)
        log_qx = bg.flow_ev_probability(bg_params, bg_static, x_bg)
        weights = bg.reweight_samples(x_bg, log_qx, T=T, potential=V, kb=kb)

        E_values = jax.vmap(V)(x_bg)
        Ess = bg.Ess(weights)
        Ess_normalized = Ess / n_samples
        E_bg_mean = jnp.sum(weights * E_values)
        var_E_weighted = jnp.sum(weights * (E_values - E_bg_mean)**2)
        E_bg_mean_err = jnp.sqrt(var_E_weighted / Ess)
        tau_eff = n_samples / (2.0 * Ess)

        results_bg["E_mean"].append(E_bg_mean)
        results_bg["E_mean_err"].append(E_bg_mean_err)
        results_bg["ESS_normalized"].append(Ess_normalized)
        results_bg["T"].append(T)
        results_bg["tau_eff"].append(tau_eff)
        results_bg["mean_sign"].append(jnp.mean(jnp.sign(x_bg[:, 0])))
        results_bg["loss_last"].append(losses[-1])
        results_bg["loss_start"].append(losses[0])

        if float(T) in T_analysis:
            pos_bg_full[float(T)] = x_bg
            logqx_bg_full[float(T)] = log_qx
            losses_bg_full[float(T)] = losses

        ##### 3. MCMC-BOLTZMANN GENERATOR ######
        x_start, key = bg.sample(bg_params, bg_static, key, n_samples=1)
        x_start = x_start[0]
        trajectory, acceptance_rate, key, x = run_flow_prod(key, T, x_start, bg_params, bg_static)

        obs.append_observables(results_mcmc_bg, T, trajectory, acceptance_rate, V,
                                tollerance, window, c, abs_tol, kb)

        if float(T) in T_analysis:
            traj_mcmc_bg_full[float(T)] = trajectory

    print("Boltzmann generator and MCMC-Boltzmann completed.")

    ####### 4. PLOTTING #######
    plot.E_vs_T_plot(results_mcmc, results_bg, results_mcmc_bg)
    plot.tau_vs_T_plot(results_mcmc, results_bg, results_mcmc_bg)
    plot.acceptance_vs_T_plot(results_mcmc, results_mcmc_bg)
    plot.mean_sign_vs_T_plot(results_mcmc, results_bg, results_mcmc_bg)
    plot.plot_loss_vs_T(results_bg)
    plot.plot_Ess_normalized_vs_T(results_mcmc, results_bg, results_mcmc_bg)
    plot.plot_loss_curves(losses_bg_full, T_analysis)

    for T in T_analysis:
        Tf = float(T)
        plot.plot_R_list_vs_m(results_mcmc, results_mcmc_bg, T)
        plot.plot_x(traj_mcmc_full[Tf], pos_bg_full[Tf], traj_mcmc_bg_full[Tf], T, V, D, kb)
        plot.plot_x_traj(traj_mcmc_full[Tf], pos_bg_full[Tf], traj_mcmc_bg_full[Tf], T)
    print("Plots for T study completed.")

    ###### 5. STUDY OF THE EFFECT OF THE NUMBER OF SAMPLES #######
    Ns = np.logspace(2, np.log10(n_samples), 15, dtype=int)

    for T in tqdm(T_analysis, desc="Studying effect of N", unit="T"):
        Tf = float(T)
        results_vsN_mcmc = vsN.res_N_mcmc(traj_mcmc_full[Tf], V, Ns, tollerance, window, c, abs_tol)
        results_vsN_bg = vsN.res_N_bg(pos_bg_full[Tf], logqx_bg_full[Tf], V, T, kb, Ns)
        results_vsN_mcmc_bg = vsN.res_N_mcmc(traj_mcmc_bg_full[Tf], V, Ns, tollerance, window, c, abs_tol)

        true_E = obs.exact_E_mean(T, V, D)
        plot.plot_E_vsN(results_vsN_mcmc, results_vsN_bg, results_vsN_mcmc_bg, T, true_E)
        plot.plot_mean_sign_vsN(results_vsN_mcmc, results_vsN_bg, results_vsN_mcmc_bg, T)
    print("Plots for N study completed.")

if __name__ == "__main__":
    main()