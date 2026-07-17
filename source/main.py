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
    T_max = 1000.0   # highest temperature
    T_min = 100.0    # lowest temperature
    T_step = 10.0     # temperature step
    Ts = jnp.arange(T_max, T_min-T_step, -T_step)  # from hot to cold
    D = 3 # dimension of the system fixed to 3 for the double well potential: 1 slow variable + 2 fast variables
    kb = 8.617333262145e-5
    key = jax.random.PRNGKey(42)
    n_samples = 1000000 # these are the number of samples for the Boltzmann generator and the steps of MCMC (10^6)
    n_train_steps = 1000 # number of training steps for the Boltzmann generator at each temperature
    n_samples_training = 500 # number of samples for the training of the Boltzmann generator at each temperature

    # potential parameters
    a = 0.1
    b = 1.0
    l = 0.0
    V = obs.make_potential(a, b, l, "simmetric_double_well")

    #results dictionaries
    results_mcmc = {
            "T": [],
            "E_mean": [],
            "E_mean_err": [],
            "acceptance": [],
            "tau_x": [],
            "mean_sign": [],
            "R_list": [],
            "trajectory": []
        } 
    
    results_bg = {
            "T": [],
            "E_mean": [],
            "E_mean_err": [],
            "ESS": [], 
            "tau_eff": [],
            "mean_sign": [],
            "loss_last": [],
            "loss_start": [],
            "ESS_normalized": [],
            "positions": [],
            "log_qx": []
        } 

    results_mcmc_bg = {
            "T": [],
            "E_mean": [],
            "E_mean_err": [],
            "acceptance": [],
            "tau_x": [],
            "mean_sign": [],
            "R_list": [],
            "trajectory": []
        }   
    
####### 0. INITIALIZATION FOR MCMC AND MCMC-BOLTZMANN #######
    tollerance = 0.1 # tollerance for the blocking analysis, to detect the plateau of R(m)
    window = 3 # number of consecutive values to consider for the plateau detection, in log scale
    c = 5 # Sokal method parameter for tau estimation, tau_int(x, c)
    abs_tol = 0.01 # absolute tolerance for plateau detection, to avoid problems with very small values of R
    n_thermalization = 2000 # number of thermalization steps for MCMC to reach equilibrium before starting the production run,
    #always thermalize, but the first thermalization is longer, to reach equilibrium from a random initial configuration
    #the other thermalizations are shorter, to reach equilibrium from the last configuration of the previsios T 
    step_size = 0.1

    gen_config = metro.make_config_generator(D, "normal")
    run_first_therm = metro.make_simulation(D, n_thermalization*10, V, kb)
    run_therm  = metro.make_simulation(D, n_thermalization, V, kb)
    run_prod   = metro.make_simulation(D, n_samples, V, kb)

    run_flow_prod = fmcmc.make_flow_simulation(n_samples, V, kb)
    
###### 1. MCMC SIMULATION ######
    x, key = gen_config(key)
    for T in tqdm(Ts, desc="MCMC", unit="T"):
        if T == Ts[0]:
            _, _, key , x = run_first_therm(key, T, step_size, x) # thermalization for the first temperature, starting from a fixed configuration (normal distribution)
        _, _, key , x = run_therm(key, T, step_size, x) # thermalization for the other temperatures, starting from the last configuration of the previous thermalization

            # x is the last configuration of the thermalization, used as initial configuration for the production
        trajectory, acceptance_rate, key , x = run_prod(key, T, step_size, x)
            # x is the last configuration of the trajectory, used as initial configuration for the next temperature

        obs.append_observables(results_mcmc, T, trajectory, acceptance_rate, V, tollerance,window, c, abs_tol, kb)

    print("MCMC simulation completed.")

###### 2. BOLTZMANN GENERATOR ######

    # --- Training del BG ---

    key, subkey = jax.random.split(key)

    bg_params, bg_static = bg.init_params(
        subkey,
        dim=D,
        n_layers=6,
        hidden_dim=32,
    )

    optimizer = optax.adam(1e-3)

    opt_state = optimizer.init(bg_params)

    train_T = bg.make_train_loop(optimizer,potential=V,kb=kb)

    for T in tqdm(Ts, desc="BG and MCMC-BG", unit="T"):
        key, subkey = jax.random.split(key)
        bg_params, opt_state, key, losses = train_T(
            bg_params, bg_static, opt_state, subkey, T,
            n_samples=n_samples_training, n_steps=n_train_steps)

        # --- Sampling e reweighting ---
        x_bg, key = bg.sample(bg_params, bg_static, key, n_samples=n_samples) 
        log_qx = bg.flow_ev_probability(bg_params, bg_static, x_bg)
        weights = bg.reweight_samples(x_bg, log_qx, T=T, potential=V, kb=kb) # pesi già normalizzati

        # Stima osservabili con pesi di importanza
        E_values = jax.vmap(V)(x_bg)
        Ess = bg.Ess(weights)
        Ess_normalized = Ess / n_samples
        E_bg_mean = jnp.sum(weights * E_values)
        var_E_weighted = jnp.sum(weights * (E_values - E_bg_mean)**2)
        E_bg_mean_err = jnp.sqrt(var_E_weighted / Ess)
        tau_eff = n_samples / (2.0 * Ess)   # stima effective sample time, 
                                            #per poter confrontare MCMC, che ha autocorrelazioni.

        results_bg["positions"].append(x_bg)
        results_bg["E_mean"].append(E_bg_mean)
        results_bg["E_mean_err"].append(E_bg_mean_err)
        results_bg["ESS"].append(Ess)
        results_bg["ESS_normalized"].append(Ess_normalized)
        results_bg["T"].append(T)
        results_bg["tau_eff"].append(tau_eff)
        results_bg["mean_sign"].append(jnp.mean(jnp.sign(x_bg[:, 0])))
        results_bg["loss_last"].append(losses[-1]) # ultimo valore della loss, per monitorare il training
        results_bg["loss_start"].append(losses[0]) # primo valore della loss, per monitorare il training
        results_bg["log_qx"].append(log_qx)

##### 3. MCMC-BOLTZMANN GENERATOR ###### we use the flow already trained (we are in the T loop of BG)
        # sampling the starting config 
        x_start, key = bg.sample(bg_params, bg_static, key, n_samples=1)                
        x_start = x_start[0]
        # running the MCMC-Boltzmann simulation
        trajectory, acceptance_rate, key, x = run_flow_prod(key, T, x_start, bg_params, bg_static) 

        obs.append_observables(results_mcmc_bg, T, trajectory, acceptance_rate, V, tollerance, window, c, abs_tol, kb)

    print("Boltzmann generator and MCMC-Boltzmann completed.")
        
####### PLOTTING #######

    plot.E_vs_T_plot(results_mcmc, results_bg, results_mcmc_bg)
    plot.tau_vs_T_plot(results_mcmc, results_bg, results_mcmc_bg)
    plot.acceptance_vs_T_plot(results_mcmc, results_mcmc_bg)
    plot.mean_sign_vs_T_plot(results_mcmc, results_bg, results_mcmc_bg)
    plot.plot_loss_vs_T(results_bg)
    plot.plot_Ess_normalized_vs_T(results_bg)
    T_R = [Ts[0], Ts[len(Ts)//2], Ts[-1]]
    for T in T_R:
        plot.plot_R_list_vs_m(results_mcmc, results_mcmc_bg, T)
        plot.plot_x(results_mcmc, results_bg, results_mcmc_bg, T)
        plot.plot_x_traj(results_mcmc, results_bg, results_mcmc_bg, T)
    print("Plots for T study completed.")

###### STUDY OF THE EFFECT OF THE NUMBER OF SAMPLES #############
    T_Ns = [Ts[0], Ts[len(Ts)//2], Ts[-1]] #hot, intermediate and cold temperatures for the study of the effect of the number of samples
    Ns = np.logspace(2,np.log10(n_samples),15,dtype=int) #number of samples from 100 to n_samples

    for T in tqdm(T_Ns, desc="Studying effect of N", unit="T"):
        idx = np.argmin(np.abs(np.array(results_mcmc["T"]) - T))
        results_vsN_mcmc = vsN.res_N_mcmc(results_mcmc["trajectory"][idx], V, Ns, tollerance, window, c, abs_tol)
        results_vsN_bg = vsN.res_N_bg(results_bg["positions"][idx], results_bg["log_qx"][idx], V, T, kb, Ns)
        results_vsN_mcmc_bg = vsN.res_N_mcmc(results_mcmc_bg["trajectory"][idx], V, Ns, tollerance, window, c, abs_tol)

        plot.plot_E_vsN(results_vsN_mcmc, results_vsN_bg, results_vsN_mcmc_bg, T)
        plot.plot_tau_vsN(results_vsN_mcmc, results_vsN_bg, results_vsN_mcmc_bg, T)

    print("Plots for N study completed.")
if __name__ == "__main__":
    main()