import mcmc as metro
import plot as plot
import observables as obs
import boltzmann_gen as bg
import optax
import jax
import jax.numpy as jnp
from tqdm import tqdm

def main():
    Ts = jnp.array([1, 10, 20, 30 ]) # temperatures to simulate
    D = 3 # dimension of the system fixed to 3 for the double well potential: 1 slow variable + 2 fast variables
    kb = 8.617333262145e-5

    key = jax.random.PRNGKey(42)

    # potential parameters
    a = 0.01
    b = 1.0
    V = obs.make_potential(a, b)

    # MCMC parameters
    n_thermalization = 1000
    n_samples = 10000 # these are the number of samples for the Boltzmann generator and the steps of MCMC
    step_size = 0.1
    tollerance = 0.10 # 10% of the mean value of the observable, in log scale
    window = 4 # number of consecutive values to consider for the plateau detection, in log scale
    c = 5 # Sokal method parameter for tau estimation, tau_int(x, c)

###### MCMC SIMULATION ######

    gen_config = metro.make_config_generator(D, "normal")
    run_first_therm = metro.make_simulation(D, n_thermalization*10, V, kb)
    run_therm  = metro.make_simulation(D, n_thermalization, V, kb)
    run_prod   = metro.make_simulation(D, n_samples, V, kb)

    results_mcmc = {
            "T": [],
            "E_mean": [],
            "E_mean_err": [],
            "acceptance": [],
            "tau_x": []
        }

    x, key = gen_config(key)
    for T in tqdm(Ts, desc="MCMC", unit="T"):
        if T == Ts[0]:
            _, _, key , x = run_first_therm(key, T, step_size, x) # thermalization for the first temperature, starting from a fixed configuration (normal distribution)
        _, _, key , x = run_therm(key, T, step_size, x) # thermalization for the other temperatures, starting from the last configuration of the previous thermalization

            # x is the last configuration of the thermalization, used as initial configuration for the production
        trajectory, acceptance_rate, key , x = run_prod(key, T, step_size, x)
            # x is the last configuration of the trajectory, used as initial configuration for the next temperature

        obs.append_observables(results_mcmc, T, trajectory, acceptance_rate, V, tollerance,window, c, kb)

    print("MCMC simulation completed.")

###### BOLTZMANN GENERATOR ######

    results_bg = {
            "T": [],
            "E_mean": [],
            "E_mean_err": [],
        }

    # --- Training del BG ---

    n_train_steps = 8000
    step_fn = bg.make_step(optimizer)

    for T in tqdm(Ts, desc="BG training", unit="T"):
        key, subkey = jax.random.split(key)
        bg_params = bg.init_params(subkey, dim=D, n_layers=6, hidden_dim=32)
        optimizer = optax.adam(learning_rate=1e-3)
        opt_state = optimizer.init(bg_params)
        for i in range(n_train_steps):
            key, subkey = jax.random.split(key)
            bg_params, opt_state, loss = step_fn(
                bg_params, opt_state, subkey, n_samples=500, T=T, a=a, b=b
            )
        # --- Sampling e reweighting ---
        x_bg, log_qx, key = bg.sample(bg_params,key, n_samples=n_samples) #dentro splitta la chiave # !!!!
        weights = bg.reweight_samples(x_bg, log_qx, T=T, a=a, b=b, kb=kb)

        # Stima osservabili con pesi di importanza
        E_bg = jnp.sum(weights * jax.vmap(V)(x_bg))
        E_bg_mean = E_bg / jnp.sum(weights)
        # Stima dell'errore standard della media pesata DA FARE!!
        results_bg["T"].append(T)
        results_bg["E_mean"].append(E_bg_mean)
        # results_bg["E_mean_err"].append(E_bg_mean_err)

    print("BG completed.")
##### MCMC-BOLTZMANN GENERATOR ######

####### PLOTTING #######




if __name__ == "__main__":
    main()