# Metodi Monte Carlo e Normalizing Flows generativi per il campionamento di distribuzioni di Boltzmann multimodali

Questo repository contiene il codice Python e l'analisi sviluppati per la tesi di laurea in Fisica presso l'Università di Bologna (A.A. 2025/2026).

- **Autore:** Riccardo Pivi
- **Relatore:** Prof. Cesare Franchini
- **Correlatore:** Dott. Luca Leoni



📄 **Manoscritto:** [Clicca qui per consultare la tesi in formato PDF](tex/thesis.pdf)
---

## 📖 Descrizione del Progetto

Il calcolo delle proprietà macroscopiche di un sistema fisico all'equilibrio termico richiede la valutazione di valori di aspettazione rispetto alla **distribuzione di Boltzmann**. Per sistemi ad alta dimensionalità o con paesaggi energetici complessi (potenziali multimodali), le tecniche di quadratura numerica tradizionale falliscono a causa della *maledizione della dimensionalità*.

I metodi stocastici come il **Markov Chain Monte Carlo (MCMC)** con proposte locali rappresentano lo standard, ma soffrono di un limite intrinseco: quando la distribuzione target presenta più modi separati da regioni a bassa densità di probabilità, la catena di Markov rimane intrappolata in un singolo modo per tempi computazionali proibitivi.

In questo progetto si esplora l'utilizzo dei **Normalizing Flows (NF)** (in particolare l'architettura **Real NVP**) per superare questo limite, confrontando tre approcci principali:
1. **MCMC Locale (Metropolis-Hastings):** Campionamento markoviano basato su proposte gaussiane locali.
2. **Normalizing Flow (Campionamento Diretto Approssimato con Re-weighting):** Generazione diretta di campioni tramite trasformazioni invertibili e differenziabili addestrate a minimizzare la divergenza di Kullback-Leibler.
3. **Flow-MCMC (Approccio Ibrido):** Algoritmo Metropolis-Hastings con mosse globali proposte dal Normalizing Flow, che unisce la capacità di esplorazione globale dei modelli generativi con la correttezza ed ergodicità asintotica di MCMC.

### Sistema di Test
Tutti i metodi vengono confrontati empiricamente su un sistema a **potenziale doppio pozzo (Double-Well Potential)**, scelto specificamente per la bimodalità indotta sulla distribuzione di Boltzmann associata, valutandone l'efficienza e la velocità diconvergenza al variare della temperatura e del numero di campioni.

---

## 📊 Risultati Principali

- **Basse Temperature:** MCMC locale rimane sistematicamente intrappolato in una sola buca del potenziale doppio pozzo, fallendo nel campionare correttamente la distribuzione di Boltzmann.
- **Normalizing Flow & Flow-MCMC:** Mantengono un'esplorazione bilanciata di entrambi i modi anche a basse temperature, garantendo tempi di autocorrelazione integrati significativamente inferiori e una convergenza più rapida ai valori di aspettazione teorici.