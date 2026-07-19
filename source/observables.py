import jax 
import jax.numpy as jnp
import numpy as np  
from typing import Callable
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.integrate import trapezoid

def make_potential(a: float = 1.0, b: float = 1.0,l: float = 0.0, potential_type: str = "simmetric_double_well") -> Callable[[jax.Array], jax.Array]:
    @jax.jit
    def V(x: jax.Array) -> jax.Array:
        if potential_type == "simmetric_double_well":
            double_well = (a/b**4) * (x[0] ** 2 - b) ** 2
            harmonic    = 0.5 * jnp.dot(x[1:], x[1:])
            return double_well + harmonic
        elif potential_type == "asymmetric_double_well":
            double_well = (a/b**4) * (x[0] ** 2 - b) ** 2 + l * x[0]
            harmonic    = 0.5 * jnp.dot(x[1:], x[1:])
            return double_well + harmonic
        else:
            raise ValueError(f"Unknown potential_type: {potential_type}, the options are: 'simmetric_double_well', 'asymmetric_double_well'")
    return V

def find_plateau(R, window=3, tollerance=0.2, abs_tol=1e-3, obs="_"): #there is a abs_tol to avoid problems with very small values of R
    window = int(window)
    R = np.array(R, dtype=float)
    logR = np.log(R)

    for k in range(window, len(R)):
        segment = logR[k-window:k]
        mean = np.mean(segment)
        std = np.std(segment)
        thresh = max(abs_tol, tollerance * abs(mean))

        if abs(logR[k] - mean) < thresh and std < thresh:
            R_plateau = R[k]
            tau = 0.5 * (R_plateau - 1)
            return R_plateau, tau

    print(f"\n WARNING: plateau not found for {obs}, using last value.")
    R_plateau = R[-1]
    tau = 0.5 * (R_plateau - 1)
    return R_plateau, tau

def blocking_analysis(data, window, threshold, abs_tol, obs="_"):
    x = jnp.array(data, dtype=jnp.float32)

    # varianza completa 
    varX = jnp.var(x, ddof=1)

    R_list = []
    cur = x.copy()
    m = 1

    # blocling diminuendo di 1 la dimensione ad ogni iterazione, fino a quando non rimangono almeno 16 blocchi
    while len(cur) >= 16:

        if cur.shape[0] % 2 == 1:
            cur = cur[:-1]

        var_block_mean = jnp.var(cur, ddof=1)

        # R(m) = m * Var(block) / Var(full)
        R = m * var_block_mean / varX
        R_list.append(R)

        # blocking → media di coppie
        cur = 0.5 * (cur[0::2] + cur[1::2])
        m *= 2
    R_plateau, tau = find_plateau(R_list,window, threshold,abs_tol, obs)
    M = data.shape[0]
    sigma_mean = jnp.sqrt(R_plateau * varX / M)
    return sigma_mean, tau, R_list

def autocorr_fft(x):
    x = x - np.mean(x)
    N = len(x)
    f = jnp.fft.fft(x, n=2*N)
    acf = jnp.fft.ifft(f * jnp.conjugate(f))[:N].real
    acf /= acf[0]
    return acf

#  tau with Sokal method
def tau_int(x, c=5):
    acf = autocorr_fft(x)
    N = len(acf)

    cum = jnp.cumsum(acf[1:])          # cum[i] = sum(acf[1:i+2])
    tau_t = 0.5 + cum                  # tau(t) per t = 1..N-1, tau_t[i] <-> t=i+1
    t_vals = jnp.arange(1, N)

    condition = t_vals > c * tau_t
    found = jnp.any(condition)
    idx = jnp.argmax(condition)        # primo True (0 se non trovato mai)

    M = jnp.where(found, t_vals[idx], N - 1)
    tau = jnp.where(found, tau_t[idx], tau_t[-1])

    if not bool(found):
        print("\n WARNING: windowing non convergente per tau_int, uso M=N-1.")

    delta_tau = tau * jnp.sqrt(2 * (2*M + 1) / len(x))
    return float(tau), float(delta_tau)

def append_observables(results, T, trajectory, acceptance_rate, V,
                        tolerance: float = 0.01, window: int = 5, c: int = 5,
                        abs_tol: float = 1e-3, kb: float = 8.617333262145e-5):
    energies = jax.vmap(V)(trajectory)    
    E_mean = float(jnp.mean(energies))
    E_mean_err, _, R_list = blocking_analysis(energies, window, tolerance, abs_tol, obs="E_mean")

    tau_x, delta_tau = tau_int(trajectory[:, 0], c)
    mean_sign = jnp.mean(jnp.sign(trajectory[:, 0]))
    Ess_normalized = 1 / (2 * tau_x)

    results["T"].append(T)
    results["E_mean"].append(E_mean)
    results["E_mean_err"].append(E_mean_err)
    results["acceptance"].append(acceptance_rate)
    results["tau_x"].append(tau_x)
    results["delta_tau"].append(delta_tau)      
    results["mean_sign"].append(mean_sign)
    results["R_list"].append(R_list)
    results["ESS_normalized"].append(Ess_normalized)

def exact_E_mean(T: float, V: Callable, dim: int,
                  kb: float = 8.617333262145e-5,
                  x_scan_range: float = 20.0, n_scan: int = 2000,
                  xlim_sigma: float = 15.0) -> float:
    """
    Valore esatto di <V> per quadratura 1D lungo x[0], usando direttamente
    la funzione V (es. quella da make_potential). Le altre (dim-1) componenti
    contribuiscono con equipartizione esatta (0.5*kb*T ciascuna), sfruttando
    il fatto che V è additivo: V(x) = V0(x0) + harmonic(x[1:]).

    dim = dimensionalità totale del vettore x.
    """
    beta = 1.0 / (kb * T)

    # sezione 1D: fissa x[1:] = 0 (harmonic(0) = 0, quindi V0(x0) = V([x0,0,...,0]))
    def V0(x0):
        x = jnp.zeros(dim).at[0].set(x0)
        return float(V(x))

    # scansione grezza per trovare il/i minimi (funziona anche per pozzo singolo)
    grid = np.linspace(-x_scan_range, x_scan_range, n_scan)
    vals = np.array([V0(x0) for x0 in grid])
    i_min = np.argmin(vals)
    x0_guess = grid[i_min]

    # rifinitura locale
    lo = grid[max(i_min-5, 0)]
    hi = grid[min(i_min+5, n_scan-1)]
    res = minimize_scalar(V0, bracket=(lo, x0_guess, hi))
    Vmin = res.fun

    # range di integrazione adattivo attorno al bulk della distribuzione
    xlim = x_scan_range  # fallback largo; si può restringere adattivamente sotto
    # stima ampiezza termica tipica per restringere xlim se serve
    xlim = min(x_scan_range, abs(x0_guess) + xlim_sigma * np.sqrt(kb * T) + 1.0)

    weight = lambda x0: np.exp(-beta * (V0(x0) - Vmin))
    num    = lambda x0: V0(x0) * weight(x0)

    Z, _     = quad(weight, -xlim, xlim, limit=400)
    num_I, _ = quad(num, -xlim, xlim, limit=400)

    V0_mean = num_I / Z
    harmonic_mean = 0.5 * (dim - 1) * kb * T

    return V0_mean + harmonic_mean

def theoretical_density(x_grid, T, V, dim, kb=8.617333262145e-5):
    """
    Calcola p(x0) marginale esatta (Boltzmann) su una griglia x_grid,
    integrando fuori le componenti veloci x[1:] (che contribuiscono
    solo con una costante di normalizzazione, essendo armoniche e
    indipendenti da x0 nel potenziale V additivo).

    Ritorna un array della stessa shape di x_grid, normalizzato
    in modo che integri a 1 su x_grid (quadratura trapezoidale).
    """
    beta = 1.0 / (kb * T)

    def V0(x0):
        x = jnp.zeros(dim).at[0].set(x0)
        return float(V(x))

    vals = np.array([V0(x0) for x0 in x_grid])
    Vmin = vals.min()  # per stabilità numerica nell'esponenziale

    unnorm = np.exp(-beta * (vals - Vmin))
    Z = trapezoid(unnorm, x_grid)  # normalizzazione sull'intervallo mostrato

    return unnorm / Z
