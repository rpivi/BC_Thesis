import mcmc as metro
import plot as plot
import observables as obs
import jax
import jax.numpy as jnp
from tqdm import tqdm

def main():
    Ts = jnp.array([1, 10, 20, 30 ]) # temperatures to simulate
    D = 3 # dimension of the system fixed to 3 for the double well potential: 1 slow variable + 2 fast variables
    kb = 8.617333262145e-5

    # potential parameters
    a = 0.01
    b = 1.0
    V = obs.make_potential(a, b)

    # MCMC parameters
    n_thermalization = 1000
    n_steps = 10000
    step_size = 0.1
    tollerance = 0.10 # 10% of the mean value of the observable, in log scale
    window = 4 # number of consecutive values to consider for the plateau detection, in log scale
    c = 5 # Sokal method parameter for tau estimation, tau_int(x, c)

###### MCMC SIMULATION ######

    gen_config = metro.make_config_generator(D, "normal")
    run_first_therm = metro.make_simulation(D, n_thermalization*10, V, kb)
    run_therm  = metro.make_simulation(D, n_thermalization, V, kb)
    run_prod   = metro.make_simulation(D, n_steps, V, kb)

    results = {
            "T": [],
            "E_mean": [],
            "E_mean_err": [],
            "acceptance": [],
            "tau_x": []
        }

    x, key = gen_config(key)
    for T in tqdm(Ts, desc=f"T={T}", unit="T"):
        if T == Ts[0]:
            _, _, key , x = run_first_therm(key, T, step_size, x) # thermalization for the first temperature, starting from a fixed configuration (normal distribution)
        _, _, key , x = run_therm(key, T, step_size, x) # thermalization for the other temperatures, starting from the last configuration of the previous thermalization

            # x is the last configuration of the thermalization, used as initial configuration for the production
        trajectory, acceptance_rate, key , x = run_prod(key, T, step_size, x)
            # x is the last configuration of the trajectory, used as initial configuration for the next temperature

        obs.append_observables(results, T, trajectory, acceptance_rate, V, tollerance,window, c, kb)

    print("MCMC simulation completed.")
    
###### BOLTZMANN GENERATOR ######

##### MCMC-BOLTZMANN GENERATOR ######

####### PLOTTING #######




if __name__ == "__main__":
    main()