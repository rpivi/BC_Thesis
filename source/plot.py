import observables as obs

from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# plot.py è in BC_Thesis/source/, quindi .parent.parent = BC_Thesis/
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "tex" / "immages"
OUT_DIR.mkdir(parents=True, exist_ok=True)
 
# Colori coerenti per metodo in TUTTI i plot della tesi.
COLORS = {
    "MCMC": "tab:blue",
    "BG": "tab:orange",
    "MCMC-BG": "tab:green",
}
MARKERS = {
    "MCMC": "o",
    "BG": "s",
    "MCMC-BG": "^",
}
 
######################## vs T #####################################
def E_vs_T_plot(results_mcmc, results_bg, results_mcmc_bg):
    """
    Plotta E_mean vs T per MCMC, Normalizing Flow e FLow-MCMC.
    """
    plt.figure(figsize=(8, 6))
    plt.errorbar(results_mcmc["T"], results_mcmc["E_mean"], yerr=results_mcmc["E_mean_err"],
                 fmt=MARKERS["MCMC"] + '-', color=COLORS["MCMC"], label='MCMC', capsize=5)
    plt.errorbar(results_bg["T"], results_bg["E_mean"], yerr=results_bg["E_mean_err"],
                 fmt=MARKERS["BG"] + '-', color=COLORS["BG"], label='Normalizing FLow', capsize=5)
    plt.errorbar(results_mcmc_bg["T"], results_mcmc_bg["E_mean"], yerr=results_mcmc_bg["E_mean_err"],
                 fmt=MARKERS["MCMC-BG"] + '-', color=COLORS["MCMC-BG"], label='FLow-MCMC', capsize=5)
 
    plt.xlabel('Temperature T [K]')
    plt.ylabel('Mean Energy <E> [eV]')
    plt.title('Mean Energy vs Temperature')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "E_vs_T_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
 
def tau_vs_T_plot(results_mcmc, results_bg, results_mcmc_bg, n_exclude_cold=5):
    """
    Plotta τ + error vs T per MCMC, Normalizing Flow e FLow-MCMC.

    La simulazione MCMC procede da T_max a T_min (hot -> cold), quindi le
    temperature più fredde sono le ultime `n_exclude_cold` voci di ciascuna
    lista in `results`. Lì la stima di tau non è affidabile (mode-trapping /
    windowing di Sokal non convergente), quindi vengono escluse dal plot.
    """
    def _trim(d):
        if n_exclude_cold <= 0:
            return d
        return {k: v[:len(v) - n_exclude_cold] for k, v in d.items()}

    r_mcmc = _trim(results_mcmc)
    r_bg = _trim(results_bg)
    r_mcmc_bg = _trim(results_mcmc_bg)

    plt.figure(figsize=(8, 6))
    plt.errorbar(r_mcmc["T"], r_mcmc["tau_x"], yerr=r_mcmc["delta_tau"],
                 fmt=MARKERS["MCMC"] + '-', color=COLORS["MCMC"], label='MCMC', capsize=5)
    plt.plot(r_bg["T"], r_bg["tau_eff"], MARKERS["BG"] + '-',
              color=COLORS["BG"], label='Normalizing Flow')
    plt.errorbar(r_mcmc_bg["T"], r_mcmc_bg["tau_x"], yerr=r_mcmc_bg["delta_tau"],
                 fmt=MARKERS["MCMC-BG"] + '-', color=COLORS["MCMC-BG"], label='FLow-MCMC', capsize=5)
    plt.yscale('symlog', linthresh=10)
    plt.xlabel('Temperature T [K]')
    plt.ylabel(r'Effective Time $\tau$')
    plt.title('Effective Time vs Temperature')
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig(OUT_DIR / "tau_vs_T_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
 
 
def acceptance_vs_T_plot(results_mcmc, results_mcmc_bg):
    """
    Plotta acceptance rate vs T per MCMC e FLow-MCMC.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(results_mcmc["T"], results_mcmc["acceptance"], MARKERS["MCMC"] + '-',
              color=COLORS["MCMC"], label='MCMC')
    plt.plot(results_mcmc_bg["T"], results_mcmc_bg["acceptance"], MARKERS["MCMC-BG"] + '-',
              color=COLORS["MCMC-BG"], label='FLow-MCMC')
 
    plt.xlabel('Temperature T [K]')
    plt.ylabel('Acceptance Rate')
    plt.ylim(0, 1)
    plt.title('Acceptance Rate vs Temperature')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "acceptance_vs_T_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
 
 
def mean_sign_vs_T_plot(results_mcmc, results_bg, results_mcmc_bg):
    """
    Plotta mean sign vs T per MCMC, Normalizing Flow e FLow-MCMC.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(results_mcmc["T"], results_mcmc["mean_sign"], MARKERS["MCMC"] + '-', color=COLORS["MCMC"], label='MCMC')
    plt.plot(results_bg["T"], results_bg["mean_sign"], MARKERS["BG"] + '-', color=COLORS["BG"], label='Normalizing Flow')
    plt.plot(results_mcmc_bg["T"], results_mcmc_bg["mean_sign"], MARKERS["MCMC-BG"] + '-', color=COLORS["MCMC-BG"], label='FLow-MCMC')
 
    plt.xlabel('Temperature T [K]')
    plt.ylabel('Mean Sign')
    plt.title('Mean Sign vs Temperature')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "mean_sign_vs_T_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
 
 
def plot_loss_vs_T(results_bg):
    """
    Plotta la loss del Normalizing Flow (start vs last) per ogni temperatura.
    """
    plt.figure(figsize=(8, 6))
    T_arr = np.asarray(results_bg["T"], dtype=float)
    plt.plot(T_arr, results_bg["loss_last"], 'o-', color=COLORS["BG"],label='Loss finale')
    plt.plot(T_arr, results_bg["loss_start"], 's--', color='tab:gray',label='Loss iniziale')

    plt.xlabel('Temperature T [K]')
    plt.ylabel('Loss')
    plt.title('Normalizing Flow Loss vs Temperature')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "bg_loss_vs_T_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
 
def plot_Ess_normalized_vs_T(results_mcmc,results_bg, results_mcmc_bg, n_exclude_cold=5):
    """
    Plotta l'Effective Sample Size (ESS) normalizzata per ogni T
    """
    def _trim(d): # escludo le stesse T che escludo per tau
            if n_exclude_cold <= 0:
                return d
            return {k: v[:len(v) - n_exclude_cold] for k, v in d.items()}
    
    r_mcmc = _trim(results_mcmc)
    r_bg = _trim(results_bg)
    r_mcmc_bg = _trim(results_mcmc_bg)
    
    plt.figure(figsize=(8, 6))
    plt.plot(r_mcmc["T"], r_mcmc["ESS_normalized"], 'o-', color=COLORS["MCMC"],
              label='MCMC')
    plt.plot(r_bg["T"], r_bg["ESS_normalized"], 'o-', color=COLORS["BG"],
              label='Normalizing Flow')
    plt.plot(r_mcmc_bg["T"], r_mcmc_bg["ESS_normalized"], 'o-', color=COLORS["MCMC-BG"],
              label='FLow-MCMC')
 
    plt.xlabel('Temperature T [K]')
    plt.ylabel('ESS normalizzata (ESS/N)')
    plt.ylim(0, 1.05)
    plt.title('ESS/N vs Temperature')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "ess_normalized_vs_T_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
 
########### Dati per T_analysis ######################
def plot_x(x_mcmc, x_bg, x_mcmc_bg, T, V, dim, kb=8.617333262145e-5):
    """
    x_mcmc, x_bg, x_mcmc_bg: array (N, D) già selezionati dal chiamante per il T richiesto.
    V, dim, kb: necessari per calcolare la densità teorica esatta da sovrapporre.
    """
    x_mcmc = x_mcmc[:, 0]
    x_bg = x_bg[:, 0]
    x_mcmc_bg = x_mcmc_bg[:, 0]

    x_min = min(x_mcmc.min(), x_bg.min(), x_mcmc_bg.min())
    x_max = max(x_mcmc.max(), x_bg.max(), x_mcmc_bg.max())
    bins = np.linspace(x_min, x_max, 51)

    # --- densità teorica esatta sulla stessa griglia visualizzata ---
    x_grid = np.linspace(x_min, x_max, 300)
    p_theory = obs.theoretical_density(x_grid, T, V, dim, kb)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True, sharey=True)
    data = [
        (x_mcmc, "MCMC", COLORS["MCMC"]),
        (x_bg, "Normalizing Flow", COLORS["BG"]),
        (x_mcmc_bg, "FLow-MCMC", COLORS["MCMC-BG"]),
    ]
    for ax, (x, label, color) in zip(axes, data):
        ax.hist(x, bins=bins, density=True, color=color, alpha=0.6, label=label)
        ax.plot(x_grid, p_theory, 'k--', linewidth=1.8, label='Teorica (Boltzmann)')
        ax.set_ylabel('Density')
        ax.set_title(label)
        ax.grid(True)
        ax.legend(loc='upper right', fontsize=9)

    axes[-1].set_xlabel('Position x[0]')
    fig.suptitle(f'Position Distribution at T={T}')
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"x_distribution_T_{T}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_x_traj(x_mcmc, x_bg, x_mcmc_bg, T, n_zoom=3000):
    """
    Colonna sinistra: zoom sui primi n_zoom step (mostra la dinamica reale,
    utile per individuare mode-trapping).
    Colonna destra: mappa di densità (hexbin) sull'intera traiettoria,
    sostituisce lo scatter illeggibile su N grandi.
    """
    x_mcmc = x_mcmc[:, 0]
    x_bg = x_bg[:, 0]
    x_mcmc_bg = x_mcmc_bg[:, 0]

    data = [
        (x_mcmc, "MCMC", COLORS["MCMC"]),
        (x_bg, "Normalizing Flow", COLORS["BG"]),
        (x_mcmc_bg, "FLow-MCMC", COLORS["MCMC-BG"]),
    ]

    y_min = min(x_mcmc.min(), x_bg.min(), x_mcmc_bg.min())
    y_max = max(x_mcmc.max(), x_bg.max(), x_mcmc_bg.max())

    fig, axes = plt.subplots(3, 2, figsize=(18, 12), sharey=True)

    for i, (x, label, color) in enumerate(data):
        ax_zoom, ax_full = axes[i, 0], axes[i, 1]

        # --- colonna sinistra: zoom primi n_zoom step ---
        n_z = min(n_zoom, len(x))
        ax_zoom.plot(np.arange(n_z), x[:n_z], color=color, lw=0.7)
        ax_zoom.set_ylabel('Position x[0]')
        ax_zoom.set_title(f'{label} — primi {n_z} step')
        ax_zoom.grid(True)

        # --- colonna destra: densità sull'intera traiettoria ---
        hb = ax_full.hexbin(
            np.arange(len(x)), x, gridsize=100, cmap='viridis',
            mincnt=1, bins='log'
        )
        ax_full.set_title(f'{label} — traiettoria completa (densità)')
        ax_full.grid(True, alpha=0.3)
        fig.colorbar(hb, ax=ax_full, label='log(conteggio)')

    for ax in axes[-1, :]:
        ax.set_xlabel('Sample Index')

    axes[0, 0].set_ylim(y_min, y_max)

    fig.suptitle(f'Position Trajectory at T={T}')
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"x_trajectory_T_{T}.png", dpi=150, bbox_inches="tight")
    plt.close()
 
def plot_R_list_vs_m(results_mcmc, results_mcmc_bg, T):
    """
    Plotta R(m) vs m per MCMC e FLow-MCMC alla temperatura T richiesta
    (blocking analysis).
    """
    idx = np.argmin(np.abs(np.array(results_mcmc["T"]) - T))
 
    R_mcmc = results_mcmc["R_list"][idx]
    R_bg = results_mcmc_bg["R_list"][idx]
 
    m_mcmc = 2 ** np.arange(len(R_mcmc))
    m_bg = 2 ** np.arange(len(R_bg))
 
    plt.figure(figsize=(8, 6))
    plt.plot(m_mcmc, R_mcmc, MARKERS["MCMC"] + '-', color=COLORS["MCMC"], label="MCMC")
    plt.plot(m_bg, R_bg, MARKERS["MCMC-BG"] + '-', color=COLORS["MCMC-BG"], label="FLow-MCMC")
 
    plt.xscale("log", base=2)
    plt.xlabel("Block size")
    plt.ylabel("R")
    plt.title(f"Blocking analysis at T={T}")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"blocking_analysis_R_vs_m_T_{T}.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_loss_curves(losses_bg_full, T_analysis):
    """
    Plotta l'andamento della loss durante il training per ciascuna T in T_analysis.
    """
    plt.figure(figsize=(8, 6))
    for T in T_analysis:
        Tf = float(T)
        losses = np.asarray(losses_bg_full[Tf])
        steps = np.arange(len(losses))
        plt.plot(steps, losses, label=f'T={Tf:.0f} K')

    plt.xlabel('Training step')
    plt.ylabel('Loss')
    plt.title('Normalizing Flow: andamento della loss durante il training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "bg_loss_curves_vs_step.png", dpi=150, bbox_inches="tight")
    plt.close()

################### vs N #############################################################
def plot_E_vsN(results_vsN_mcmc, results_vsN_bg, results_vsN_mcmc_bg, T, true_E,
               zoom_n_points=5, zoom_loc=None, zoom_size="42%"):
    """
    Plotta E_mean ± errore come banda colorata, con un inset che zooma
    sugli ultimi `zoom_n_points` valori di N (la coda ad alto N, dove si
    apprezza la convergenza al valore vero).

    Se zoom_loc è None, la posizione dell'inset (in alto o in basso a destra)
    viene scelta automaticamente in base a dove i dati lasciano più spazio
    libero, così l'inset non si sovrappone mai alle curve.
    """

    Ns = np.asarray(results_vsN_mcmc["N"])

    methods = [("MCMC", results_vsN_mcmc), ("BG", results_vsN_bg), ("MCMC-BG", results_vsN_mcmc_bg)]
    labels = {"MCMC": "MCMC", "BG": "Normalizing Flow", "MCMC-BG": "FLow-MCMC"}

    fig, ax = plt.subplots(figsize=(8, 6))

    all_E = {}
    all_err = {}

    for method, results in methods:
        E = np.asarray(results["E_mean"])
        err = np.asarray(results["E_mean_err"])
        all_E[method] = E
        all_err[method] = err

        ax.plot(Ns, E, color=COLORS[method], linestyle='-', linewidth=2, label=labels[method])
        ax.fill_between(Ns, E - err, E + err, color=COLORS[method], alpha=0.25)

    ax.axhline(y=true_E, color='k', linestyle='--', label=f'True E: {true_E}')

    ax.set_xscale("log")
    ax.set_xlabel("Number of Samples (N)")
    ax.set_ylabel(r"Mean Energy $\langle E\rangle$  [eV]")
    ax.set_title(f"Mean Energy vs Number of Samples at T={T}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # --- scelta automatica della posizione dell'inset ---
    if zoom_loc is None:
        y_bottom, y_top = ax.get_ylim()
        y_mid = 0.5 * (y_bottom + y_top)

        # frazione della porzione DESTRA del grafico (ultimo 30% in N)
        # che sta sopra il centro verticale -> decide se mettere l'inset
        # in alto o in basso, cosi non copre mai le curve
        n_right = max(1, int(0.3 * len(Ns)))
        right_vals = np.concatenate([all_E[m][-n_right:] for m, _ in methods])
        frac_right_above_mid = np.mean(right_vals > y_mid)

        zoom_loc = "lower right" if frac_right_above_mid > 0.5 else "upper right"

    # --- creazione inset ---
    axins = inset_axes(ax, width=zoom_size, height=zoom_size, loc=zoom_loc)

    for method, results in methods:
        E = all_E[method]
        err = all_err[method]
        axins.plot(Ns, E, color=COLORS[method], linestyle='-', linewidth=1.5)
        axins.fill_between(Ns, E - err, E + err, color=COLORS[method], alpha=0.25)

    axins.axhline(y=true_E, color='k', linestyle='--')

    Ns_zoom = Ns[-zoom_n_points:]
    E_zoom_vals = np.concatenate([all_E[m][-zoom_n_points:] for m, _ in methods])
    err_zoom_vals = np.concatenate([all_err[m][-zoom_n_points:] for m, _ in methods])

    x_min, x_max = Ns_zoom.min(), Ns_zoom.max()
    y_min = (E_zoom_vals - err_zoom_vals).min()
    y_max = (E_zoom_vals + err_zoom_vals).max()
    y_pad = 0.1 * (y_max - y_min) if y_max > y_min else 0.1 * abs(y_max)

    axins.set_xlim(x_min * 0.9, x_max * 1.1)
    axins.set_ylim(y_min - y_pad, y_max + y_pad)
    axins.set_xscale("log")
    axins.grid(True, which="both", alpha=0.3)

    # loc1/loc2 di mark_inset vanno invertiti a seconda che l'inset
    # sia in basso (2,4) o in alto (3,1), altrimenti le linee tratteggiate
    # di collegamento risultano storte
    if zoom_loc == "lower right":
        loc1, loc2 = 2, 4
    else:
        loc1, loc2 = 3, 1

    mark_inset(ax, axins, loc1=loc1, loc2=loc2, fc="none", ec="0.5", linestyle=":")

    plt.tight_layout()
    plt.savefig(OUT_DIR / f"E_vs_N_plot_T_{T}.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_mean_sign_vsN(results_vsN_mcmc, results_vsN_bg, results_vsN_mcmc_bg, T):
    """
    Plotta mean_sign vs N per MCMC, Normalizing Flow e FLow-MCMC.
    """
    Ns = results_vsN_mcmc["N"]
 
    plt.figure(figsize=(8, 6))
    plt.plot(Ns, results_vsN_mcmc["mean_sign"], MARKERS["MCMC"] + '-', color=COLORS["MCMC"], label='MCMC')
    plt.plot(Ns, results_vsN_bg["mean_sign"], MARKERS["BG"] + '-',color=COLORS["BG"], label='Normalizing Flow')
    plt.plot(Ns, results_vsN_mcmc_bg["mean_sign"], MARKERS["MCMC-BG"] + '-', color=COLORS["MCMC-BG"], label='FLow-MCMC')
    plt.axhline(y=0, color='k', linestyle='--', label='True Mean Sign = 0')
 
    plt.xscale("log")
    plt.xlabel('Number of Samples (N)')
    plt.ylabel(r'Sign of Mean Position $\langle \text{sgn}(x_0) \rangle$')
    plt.title(f'Sign of Mean Position vs Number of Samples at T={T}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"mean_sign_vs_N_plot_T_{T}.png", dpi=150, bbox_inches="tight")
    plt.close()