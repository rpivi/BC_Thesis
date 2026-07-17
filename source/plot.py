from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np

# plot.py is in BC_Thesis/source/, quindi .parent.parent = BC_Thesis/
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "tex" / "immages"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def E_vs_T_plot(results_mcmc, results_bg, results_mcmc_bg):
    """
    Plotta E_mean vs T per MCMC, Boltzmann Generator e MCMC-Boltzmann.
    """
    plt.figure(figsize=(8, 6))
    plt.errorbar(results_mcmc["T"], results_mcmc["E_mean"], yerr=results_mcmc["E_mean_err"], fmt='o-', label='MCMC', capsize=5)
    plt.errorbar(results_bg["T"], results_bg["E_mean"], yerr=results_bg["E_mean_err"], fmt='s-', label='Boltzmann Generator', capsize=5)
    plt.errorbar(results_mcmc_bg["T"], results_mcmc_bg["E_mean"], yerr=results_mcmc_bg["E_mean_err"], fmt='^-', label='MCMC-Boltzmann', capsize=5)
    
    plt.xlabel('Temperature (T)')
    plt.ylabel('Mean Energy <E>')
    plt.title('Mean Energy vs Temperature')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #save the plot in the tex/images folder outside the source folder
    plt.savefig(OUT_DIR / "E_vs_T_plot.png", dpi=150, bbox_inches="tight")

def tau_vs_T_plot(results_mcmc, results_bg, results_mcmc_bg):
    """
    Plotta τ vs T per MCMC, Boltzmann Generator e MCMC-Boltzmann.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(results_mcmc["T"], results_mcmc["tau_x"], 'o-', label='MCMC')
    plt.plot(results_bg["T"], results_bg["tau_eff"], 's-', label='Boltzmann Generator')
    plt.plot(results_mcmc_bg["T"], results_mcmc_bg["tau_x"], '^-', label='MCMC-Boltzmann')

    plt.xlabel('Temperature (T)')
    plt.ylabel('Effective Time τ')
    plt.title('Effective Time vs Temperature')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #save the plot in the tex/images folder outside the source folder
    plt.savefig(OUT_DIR / "tau_vs_T_plot.png", dpi=150, bbox_inches="tight")

def acceptance_vs_T_plot(results_mcmc, results_mcmc_bg):
    """
    Plotta acceptance rate vs T per MCMC, Boltzmann Generator e MCMC-Boltzmann.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(results_mcmc["T"], results_mcmc["acceptance"], 'o-', label='MCMC')
    plt.plot(results_mcmc_bg["T"], results_mcmc_bg["acceptance"], '^-', label='MCMC-Boltzmann')

    plt.xlabel('Temperature (T)')
    plt.ylabel('Acceptance Rate')
    plt.title('Acceptance Rate vs Temperature')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #save the plot in the tex/images folder outside the source folder
    plt.savefig(OUT_DIR / "acceptance_vs_T_plot.png", dpi=150, bbox_inches="tight")

def total_sign_vs_T_plot(results_mcmc, results_bg, results_mcmc_bg):
    """
    Plotta total sign vs T per MCMC, Boltzmann Generator e MCMC-Boltzmann.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(results_mcmc["T"], results_mcmc["total_sign"], 'o-', label='MCMC')
    plt.plot(results_bg["T"], results_bg["total_sign"], 's-', label='Boltzmann Generator')
    plt.plot(results_mcmc_bg["T"], results_mcmc_bg["total_sign"], '^-', label='MCMC-Boltzmann')

    plt.xlabel('Temperature (T)')
    plt.ylabel('Total Sign')
    plt.title('Total Sign vs Temperature')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #save the plot in the tex/images folder outside the source folder
    plt.savefig(OUT_DIR / "total_sign_vs_T_plot.png", dpi=150, bbox_inches="tight")

def plot_loss_vs_T(results_bg):
    """
    Plotta la loss del Boltzmann Generator durante il training per ogni temperatura.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(results_bg["T"], results_bg["loss_last"], 'o-', label='Boltzmann Generator Loss Last')
    plt.plot(results_bg["T"], results_bg["loss_start"], 's--', label='Boltzmann Generator Loss Start')
    
    plt.xlabel('Temperature (T)')
    plt.ylabel('Last Loss')
    plt.title('Boltzmann Generator Loss vs Temperature')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #save the plot in the tex/images folder outside the source folder
    plt.savefig(OUT_DIR / "bg_loss_vs_T_plot.png", dpi=150, bbox_inches="tight")

def plot_Ess_normalized_vs_T(results_bg):
    """
    Plotta l'Effective Sample Size (ESS) del Boltzmann Generator per ogni temperatura.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(results_bg["T"], results_bg["ESS_normalized"], 's--', label='Boltzmann Generator ESS Normalized')
    
    plt.xlabel('Temperature (T)')
    plt.ylabel('Effective Sample Size (ESS)')
    plt.title('Boltzmann Generator ESS vs Temperature')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #save the plot in the tex/images folder outside the source folder
    plt.savefig(OUT_DIR / "bg_ess_normalized_vs_T_plot.png", dpi=150, bbox_inches="tight")

def plot_R_list_vs_m(results_mcmc, results_mcmc_bg, T):
    """
    Plotta R(m) vs m per MCMC e MCMC-Boltzmann
    alla temperatura T richiesta.
    """

    # trova indice della temperatura
    idx_mcmc = np.argmin(np.abs(np.array(results_mcmc["T"]) - T))
    idx_bg = np.argmin(np.abs(np.array(results_mcmc_bg["T"]) - T))

    # estrai R_list della temperatura scelta
    R_mcmc = results_mcmc["R_list"][idx_mcmc]
    R_bg = results_mcmc_bg["R_list"][idx_bg]

    plt.figure(figsize=(8, 6))

    # vero block size
    m_mcmc = 2 ** np.arange(len(R_mcmc))
    m_bg = 2 ** np.arange(len(R_bg))

    plt.plot(
        m_mcmc,
        R_mcmc,
        'o-',
        label="MCMC"
    )

    plt.plot(
        m_bg,
        R_bg,
        '^-',
        label="MCMC-Boltzmann"
    )

    plt.xscale("log", base=2)

    plt.xlabel("Block size m")
    plt.ylabel("R(m)")
    plt.title(f"Blocking analysis at T={T}")

    plt.legend()
    plt.grid(True, which="both")

    plt.tight_layout()

    plt.savefig(
        OUT_DIR / f"blocking_analysis_R_vs_m_T_{T}.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()