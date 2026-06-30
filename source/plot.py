from matplotlib import pyplot as plt
from pathlib import Path

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