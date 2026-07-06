import jax 
import jax.numpy as jnp
import numpy as np  
from typing import Callable

def make_potential(a: float = 1.0, b: float = 1.0,c: float = 0.0, potential_type: str = "simmetric_double_well") -> Callable[[jax.Array], jax.Array]:
    @jax.jit
    def V(x: jax.Array) -> jax.Array:
        if potential_type == "simmetric_double_well":
            double_well = (a/b**4) * (x[0] ** 2 - b) ** 2
            harmonic    = 0.5 * jnp.dot(x[1:], x[1:])
            return double_well + harmonic
        elif potential_type == "asymmetric_double_well":
            double_well = (a/b**4) * (x[0] ** 2 - b) ** 2 + c * x[0]
            harmonic    = 0.5 * jnp.dot(x[1:], x[1:])
            return double_well + harmonic
    return V

def find_plateau(R, window=3, tolerance=0.2, obs="_"):
    window = int(window)
    R = np.array(R, dtype=float)

    # we use log because R can grow esponenzialmente, and we want to detect plateaus in a more stable way
    logR = np.log(R)

    for k in range(window, len(R)):

        # 1. Window of previous values in log scale
        segment = logR[k-window:k]

        # 2. local mean in log scale
        mean = np.mean(segment)

        # 3. STD of log values in the window
        std = np.std(segment)

        # 4. Check of plateau:
        #    Last values should be close to the local mean, within tolerance * mean (in log scale)
        if abs(logR[k] - mean) < tolerance * abs(mean) and std < tolerance * abs(mean):
            # we want the linear value of R, not the log
            R_plateau = R[k]
            tau = 0.5 * (R_plateau - 1)
            return R_plateau, tau

    # --- Fall-back: plateau not found ---
    print(f"WARNING: plateau not found for {obs}, using last value.")
    R_plateau = R[-1]
    tau = 0.5 * (R_plateau - 1)
    return R_plateau, tau

def blocking_analysis(data, window, threshold, obs="_"):
    x = jnp.array(data, dtype=jnp.float32)

    # varianza completa 
    varX = jnp.var(x, ddof=1)

    R_list = []
    cur = x.copy()
    m = 1

    # blocling diminuendo di 1 la dimensione ad ogni iterazione, fino a quando non rimangono almeno 2 blocchi
    while cur.shape[0] >= 2:

        if cur.shape[0] % 2 == 1:
            cur = cur[:-1]

        var_block_mean = jnp.var(cur, ddof=1)

        # R(m) = m * Var(block) / Var(full)
        R = m * var_block_mean / varX
        R_list.append(R)

        # blocking → media di coppie
        cur = 0.5 * (cur[0::2] + cur[1::2])
        m *= 2
    R_plateau, tau = find_plateau(R_list,window, threshold, obs)
    M = data.shape[0]
    sigma_mean = jnp.sqrt(R_plateau * varX / M)
    return sigma_mean, tau

def autocorr_fft(x):
    x = x - np.mean(x)
    N = len(x)
    f = np.fft.fft(x, n=2*N)
    acf = np.fft.ifft(f * np.conjugate(f))[:N].real
    acf /= acf[0]
    return acf

#  tau with Sokal method
def tau_int(x, c=5):
    acf = autocorr_fft(x)
    N = len(acf)
    tau = 0.5
    window = 1
    for t in range(1, N):
        tau += acf[t]
        if t > c * tau:
            break
    return tau

def append_observables(results,T,trajectory,acceptance_rate,V: Callable,tolerance: float = 0.01, window: int = 5,c: int = 5, kb: float = 8.617333262145e-5):
    energies = np.array(jax.vmap(lambda x: V(x))(trajectory))
    energies2 = energies**2

    # blocking for E
    E_mean =np.mean(energies)
    E_mean_err, _ = blocking_analysis(energies, window, tolerance, obs="E_mean")

    # tau for x[0]
    tau_x = tau_int(trajectory[:, 0],c)

    results["T"].append(T)
    results["E_mean"].append(E_mean)
    results["E_mean_err"].append(E_mean_err)
    results["acceptance"].append(acceptance_rate)
    results["tau_x"].append(tau_x)
