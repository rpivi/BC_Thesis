import observables as obs

from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
 
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
    Plotta E_mean vs T per MCMC, Boltzmann Generator e MCMC-Boltzmann.
    """
    plt.figure(figsize=(8, 6))
    plt.errorbar(results_mcmc["T"], results_mcmc["E_mean"], yerr=results_mcmc["E_mean_err"],
                 fmt=MARKERS["MCMC"] + '-', color=COLORS["MCMC"], label='MCMC', capsize=5)
    plt.errorbar(results_bg["T"], results_bg["E_mean"], yerr=results_bg["E_mean_err"],
                 fmt=MARKERS["BG"] + '-', color=COLORS["BG"], label='Boltzmann Generator', capsize=5)
    plt.errorbar(results_mcmc_bg["T"], results_mcmc_bg["E_mean"], yerr=results_mcmc_bg["E_mean_err"],
                 fmt=MARKERS["MCMC-BG"] + '-', color=COLORS["MCMC-BG"], label='MCMC-Boltzmann', capsize=5)
 
    plt.xlabel('Temperature T [K]')
    plt.ylabel('Mean Energy <E> [eV]')
    plt.title('Mean Energy vs Temperature')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "E_vs_T_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
 
def tau_vs_T_plot(results_mcmc, results_bg, results_mcmc_bg):
    """
    Plotta τ + error vs T per MCMC, Boltzmann Generator e MCMC-Boltzmann.
    """
    plt.figure(figsize=(8, 6))
    plt.errorbar(results_mcmc["T"], results_mcmc["tau_x"], yerr=results_mcmc["delta_tau"],
                 fmt=MARKERS["MCMC"] + '-', color=COLORS["MCMC"], label='MCMC', capsize=5)
    plt.plot(results_bg["T"], results_bg["tau_eff"], MARKERS["BG"] + '-',
              color=COLORS["BG"], label='Boltzmann Generator')
    plt.errorbar(results_mcmc_bg["T"], results_mcmc_bg["tau_x"], yerr=results_mcmc_bg["delta_tau"],
                 fmt=MARKERS["MCMC-BG"] + '-', color=COLORS["MCMC-BG"], label='MCMC-Boltzmann', capsize=5)
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
    Plotta acceptance rate vs T per MCMC e MCMC-Boltzmann.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(results_mcmc["T"], results_mcmc["acceptance"], MARKERS["MCMC"] + '-',
              color=COLORS["MCMC"], label='MCMC')
    plt.plot(results_mcmc_bg["T"], results_mcmc_bg["acceptance"], MARKERS["MCMC-BG"] + '-',
              color=COLORS["MCMC-BG"], label='MCMC-Boltzmann')
 
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
    Plotta mean sign vs T per MCMC, Boltzmann Generator e MCMC-Boltzmann.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(results_mcmc["T"], results_mcmc["mean_sign"], MARKERS["MCMC"] + '-', color=COLORS["MCMC"], label='MCMC')
    plt.plot(results_bg["T"], results_bg["mean_sign"], MARKERS["BG"] + '-', color=COLORS["BG"], label='Boltzmann Generator')
    plt.plot(results_mcmc_bg["T"], results_mcmc_bg["mean_sign"], MARKERS["MCMC-BG"] + '-', color=COLORS["MCMC-BG"], label='MCMC-Boltzmann')
 
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
    Plotta la loss del Boltzmann Generator (start vs last) per ogni temperatura.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(results_bg["T"], results_bg["loss_last"], 'o-', color=COLORS["BG"],label='Loss finale')
    plt.plot(results_bg["T"], results_bg["loss_start"], 's--', color='tab:gray',label='Loss iniziale')
 
    plt.xlabel('Temperature T [K]')
    plt.ylabel('Loss')
    plt.title('Boltzmann Generator Loss vs Temperature')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "bg_loss_vs_T_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
 
 
def plot_Ess_normalized_vs_T(results_mcmc,results_bg, results_mcmc_bg):
    """
    Plotta l'Effective Sample Size (ESS) normalizzata del Boltzmann Generator per ogni T.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(results_mcmc["T"], results_mcmc["ESS_normalized"], 'o-', color=COLORS["MCMC"],
              label='MCMC')
    plt.plot(results_bg["T"], results_bg["ESS_normalized"], 'o-', color=COLORS["BG"],
              label='Boltzmann Generator')
    plt.plot(results_mcmc_bg["T"], results_mcmc_bg["ESS_normalized"], 'o-', color=COLORS["MCMC-BG"],
              label='MCMC-Boltzmann')
 
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
        (x_bg, "Boltzmann Generator", COLORS["BG"]),
        (x_mcmc_bg, "MCMC-Boltzmann", COLORS["MCMC-BG"]),
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
        (x_bg, "Boltzmann Generator", COLORS["BG"]),
        (x_mcmc_bg, "MCMC-Boltzmann", COLORS["MCMC-BG"]),
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
    Plotta R(m) vs m per MCMC e MCMC-Boltzmann alla temperatura T richiesta
    (blocking analysis).
    """
    idx = np.argmin(np.abs(np.array(results_mcmc["T"]) - T))
 
    R_mcmc = results_mcmc["R_list"][idx]
    R_bg = results_mcmc_bg["R_list"][idx]
 
    m_mcmc = 2 ** np.arange(len(R_mcmc))
    m_bg = 2 ** np.arange(len(R_bg))
 
    plt.figure(figsize=(8, 6))
    plt.plot(m_mcmc, R_mcmc, MARKERS["MCMC"] + '-', color=COLORS["MCMC"], label="MCMC")
    plt.plot(m_bg, R_bg, MARKERS["MCMC-BG"] + '-', color=COLORS["MCMC-BG"], label="MCMC-Boltzmann")
 
    plt.xscale("log", base=2)
    plt.xlabel("Block size m")
    plt.ylabel("R(m)")
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
    plt.title('Boltzmann Generator: andamento della loss durante il training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "bg_loss_curves_vs_step.png", dpi=150, bbox_inches="tight")
    plt.close()

################### vs N #############################################################

def plot_E_vsN(results_vsN_mcmc, results_vsN_bg, results_vsN_mcmc_bg, T, true_E):
    """
    Plotta E_mean ± errore come banda colorata.
    """

    Ns = results_vsN_mcmc["N"]

    plt.figure(figsize=(8, 6))

    methods = [("MCMC", results_vsN_mcmc),("BG", results_vsN_bg),("MCMC-BG", results_vsN_mcmc_bg)]

    labels = {"MCMC": "MCMC","BG": "Boltzmann Generator","MCMC-BG": "MCMC-Boltzmann"}

    for method, results in methods:

        E = np.asarray(results["E_mean"])
        err = np.asarray(results["E_mean_err"])

        plt.plot( Ns, E, color=COLORS[method], linestyle='-', linewidth=2, label=labels[method])

        plt.fill_between(Ns,E - err,E + err,color=COLORS[method],alpha=0.25)
    #true value of E at this T 
    plt.axhline(y=true_E, color='k', linestyle='--', label='True E')

    plt.xscale("log")
    plt.xlabel("Number of Samples (N)")
    plt.ylabel(r"Mean Energy $\langle E\rangle$  [eV]")
    plt.title(f"Mean Energy vs Number of Samples at T={T}")

    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"E_vs_N_plot_T_{T}.png",dpi=150,bbox_inches="tight" )
    plt.close()

def plot_mean_sign_vsN(results_vsN_mcmc, results_vsN_bg, results_vsN_mcmc_bg, T):
    """
    Plotta mean_sign vs N per MCMC, Boltzmann Generator e MCMC-Boltzmann.
    """
    Ns = results_vsN_mcmc["N"]
 
    plt.figure(figsize=(8, 6))
    plt.plot(Ns, results_vsN_mcmc["mean_sign"], MARKERS["MCMC"] + '-', color=COLORS["MCMC"], label='MCMC')
    plt.plot(Ns, results_vsN_bg["mean_sign"], MARKERS["BG"] + '-',color=COLORS["BG"], label='Boltzmann Generator')
    plt.plot(Ns, results_vsN_mcmc_bg["mean_sign"], MARKERS["MCMC-BG"] + '-', color=COLORS["MCMC-BG"], label='MCMC-Boltzmann')
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