import mcmc as metro
import boltzmann_gen as bg
import mcmc_boltzmann as fmcmc
import plot as plot
import observables as obs
import optax
import jax
import jax.numpy as jnp
from tqdm import tqdm

def main():
    # Simulation parameters
    T_max = 1000.0   # temperatura più alta
    T_min = 10.0     # temperatura più bassa
    Ts = jnp.arange(T_max, T_min ,-100)  # from hot to cold
    D = 3 # dimension of the system fixed to 3 for the double well potential: 1 slow variable + 2 fast variables
    kb = 8.617333262145e-5
    key = jax.random.PRNGKey(42)
    n_samples = 10000 # these are the number of samples for the Boltzmann generator and the steps of MCMC
    n_train_steps = 8000 # number of training steps for the Boltzmann generator at each temperature

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
            "total_sign": []
        } 
    
    results_bg = {
            "T": [],
            "E_mean": [],
            "E_mean_err": [],
            "ESS": [], 
            "tau_eff": [],
            "total_sign": []
        } 

    results_mcmc_bg = {
            "T": [],
            "E_mean": [],
            "E_mean_err": [],
            "acceptance": [],
            "tau_x": [],
            "total_sign": []
        }   
    
####### 0. INITIALIZATION FOR MCMC AND MCMC-BOLTZMANN #######
    tollerance = 0.1
    window = 5 # number of consecutive values to consider for the plateau detection, in log scale
    c = 5 # Sokal method parameter for tau estimation, tau_int(x, c)
    abs_tol = 0.01 # absolute tolerance for plateau detection, to avoid problems with very small values of R
    n_thermalization = 1000
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
            n_samples=500, n_steps=n_train_steps)

        # --- Sampling e reweighting ---
        x_bg, key = bg.sample(bg_params, bg_static, key, n_samples=n_samples) 
        log_qx = bg.flow_ev_probability(bg_params, bg_static, x_bg)
        weights = bg.reweight_samples(x_bg, log_qx, T=T, potential=V, kb=kb) # pesi già normalizzati

        # Stima osservabili con pesi di importanza
        E_values = jax.vmap(V)(x_bg)
        Ess = bg.Ess(weights)
        E_bg_mean = jnp.sum(weights * E_values)
        var_E_weighted = jnp.sum(weights * (E_values - E_bg_mean)**2)
        E_bg_mean_err = jnp.sqrt(var_E_weighted / Ess)
        tau_eff = n_samples / (2.0 * Ess)   # stima effective sample time, 
                                            #per poter confrontare MCMC, che ha autocorrelazioni.

        results_bg["E_mean"].append(E_bg_mean)
        results_bg["E_mean_err"].append(E_bg_mean_err)
        results_bg["ESS"].append(Ess)
        results_bg["T"].append(T)
        results_bg["tau_eff"].append(tau_eff)
        results_bg["total_sign"].append(jnp.sum(jnp.sign(x_bg[:, 0])))
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
    plot.total_sign_vs_T_plot(results_mcmc, results_bg, results_mcmc_bg)
    print("All simulations completed and plot saved.")

if __name__ == "__main__":
    main()