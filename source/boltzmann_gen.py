import jax
import jax.numpy as jnp
import optax

# ---------------------------------------------------------------------------
# MLP 
# ---------------------------------------------------------------------------

def _mlp_init(key, in_dim, hidden_dim, out_dim):
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "W1": jax.random.normal(k1, (in_dim, hidden_dim)) * jnp.sqrt(1.0 / in_dim),
        "b1": jnp.zeros(hidden_dim),
        "W2": jax.random.normal(k2, (hidden_dim, hidden_dim)) * jnp.sqrt(1.0 / hidden_dim),
        "b2": jnp.zeros(hidden_dim),
        "W3": jax.random.normal(k3, (hidden_dim, out_dim)) * 0.01,  # ultimo layer piccolo in modo che la rete inizi vicino a zero (diventi una identità)        
        "b3": jnp.zeros(out_dim),
    }


def _mlp(params, x):
    """MLP a 2 hidden layers con attivazione tanh."""
    h = jnp.tanh(x @ params["W1"] + params["b1"])
    h = jnp.tanh(h @ params["W2"] + params["b2"])
    return h @ params["W3"] + params["b3"]


# ---------------------------------------------------------------------------
# Coupling layer RealNVP 
# ---------------------------------------------------------------------------
# Dati d dimensioni totali, la coupling layer:
#   - passa le prime d_pass dimensioni invariate  (x1 = z1)
#   - trasforma le restanti d_transform con s,t  (x2 = z2 * exp(s(z1)) + t(z1))
#
# Uso:
#   s_out = scale_factor * tanh(mlp_s(z1))
# con scale_factor ~ 2.0 per evitare esplosioni di exp(s).

_SCALE_FACTOR = 2.0


def _coupling_forward(s_params, t_params, z, d_pass):
    """
    Forward di una singola coupling layer.
    d_pass: numero di dimensioni passate invariate (le altre vengono trasformate).
    Ritorna (x, log_det_jacobian) con log_det shape (batch,).
    """
    z1 = z[:, :d_pass]        # (batch, d_pass)
    z2 = z[:, d_pass:]        # (batch, d - d_pass)

    s_raw = _mlp(s_params, z1)                         # (batch, d - d_pass)
    s = _SCALE_FACTOR * jnp.tanh(s_raw)               # bounded: evita exp overflow
    t = _mlp(t_params, z1)                             # (batch, d - d_pass)

    x1 = z1
    x2 = z2 * jnp.exp(s) + t

    log_det = jnp.sum(s, axis=1)                       # (batch,)
    x = jnp.concatenate([x1, x2], axis=1)
    return x, log_det

def _coupling_inverse(s_params, t_params, x, d_pass):
    """
    Inversa esatta della coupling layer forward.
    """
    x1 = x[:, :d_pass]
    x2 = x[:, d_pass:]

    s_raw = _mlp(s_params, x1)
    s = _SCALE_FACTOR * jnp.tanh(s_raw)
    t = _mlp(t_params, x1)

    z1 = x1
    z2 = (x2 - t) * jnp.exp(-s)

    log_det = -jnp.sum(s, axis=1)                      # segno negativo per l'inversa
    z = jnp.concatenate([z1, z2], axis=1)
    return z, log_det

def _permute(x):
    """Inverte l'ordine delle dimensioni. È la sua stessa inversa."""
    return x[:, ::-1]


# ---------------------------------------------------------------------------
# Inizializzazione parametri
# ---------------------------------------------------------------------------

def init_params(key, dim=2, n_layers=6, hidden_dim=32):
    """
    Inizializza i parametri per n_layers coupling layers.

    dim: dimensionalità dello spazio (2 o 3).
    Per dim=2: d_pass alterna tra 1 e 1.
    Per dim=3: d_pass alterna tra 1 e 2 (prima passa 1, poi dopo permute passa 2).

    Ritorna un dict con chiavi "layers": lista di {"s": ..., "t": ...}.
    """
    keys = jax.random.split(key, 2 * n_layers)
    layers = []
    for i in range(n_layers):
        # d_pass per questo layer (prima della permutazione eventuale)
        # Usiamo sempre d_pass=1 (la variabile lenta) come input alle reti s,t.
        # Dopo la permutazione le dimensioni si scambiano, quindi il layer successivo
        # riceverà le variabili veloci come input invariato. Questo alterna naturalmente.
        #
        # Per generalità: d_pass_i = 1 se i pari, (dim-1) se i dispari
        # (prima di qualsiasi permutazione; la permutazione è applicata *dopo* ogni layer)
        if i % 2 == 0:
            d_pass_i = 1
        else:
            d_pass_i = dim - 1

        d_transform_i = dim - d_pass_i

        layers.append({
            "s": _mlp_init(keys[2 * i], d_pass_i, hidden_dim, d_transform_i),
            "t": _mlp_init(keys[2 * i + 1], d_pass_i, hidden_dim, d_transform_i),
            "d_pass": d_pass_i,
        })

    return {"layers": layers, "dim": dim}

# ---------------------------------------------------------------------------
# Forward e inversa
# ---------------------------------------------------------------------------

def forward(params, z):
    """
    Mappa z ~ N(0,I) -> x ~ q_flow(x).
    Ritorna (x, log_det_totale) con log_det shape (batch,).
    """
    x = z
    log_det_total = jnp.zeros(z.shape[0])

    for layer in params["layers"]:
        x, ld = _coupling_forward(layer["s"], layer["t"], x, layer["d_pass"])
        log_det_total += ld
        x = _permute(x)

    return x, log_det_total


def inverse(params, x):
    """
    Mappa x -> z (inversa esatta).
    Ritorna (z, log_det_totale_inverso).
    """
    z = x
    log_det_total = jnp.zeros(x.shape[0])

    # L'inversa percorre i layer al contrario, con permutazione *prima* di ogni layer
    for layer in reversed(params["layers"]):
        z = _permute(z)                        # inversa della permutazione (che è se stessa)
        z, ld = _coupling_inverse(layer["s"], layer["t"], z, layer["d_pass"])
        log_det_total += ld

    return z, log_det_total


# ---------------------------------------------------------------------------
# Log-probabilità
# ---------------------------------------------------------------------------

def log_pz(z):
    """Log-probabilità della base distribution N(0, I). Shape: (batch,)."""
    d = z.shape[1]
    return -0.5 * jnp.sum(z ** 2, axis=1) - 0.5 * d * jnp.log(2.0 * jnp.pi)


def log_q_target(x, T, a=1.0, b=1.0, kb=8.617333262145e-5):
    """
    p(x) ∝ exp(-E(x) / (kb * T))
    E(x) = a*(x_slow^2 - b)^2 + 0.5 * sum(x_fast^2)
    """
    x_slow = x[:, 0]
    x_fast = x[:, 1:]

    energy = (a/b**4) * (x_slow ** 2 - b) ** 2 + 0.5 * jnp.sum(x_fast ** 2, axis=1) # stesso potenziale di observable.py

    return -energy / (kb * T)


# ---------------------------------------------------------------------------
# Loss KL forward
# ---------------------------------------------------------------------------

def loss_fn(params, key, n_samples=500, T=1.0, a=1.0, b=1.0):
    """
    Minimizza KL(q_flow || p_target):
        L = E_{z~N}[ log p_z(z) - log_det J - log p_target(x) ]
          = E_{z~N}[ log p_z(z) - log_det J - log_q_target(f(z)) ]

    Questo è equivalente a minimizzare -ELBO (evidence lower bound).
    """
    dim = params["dim"]
    z = jax.random.normal(key, (n_samples, dim))
    x, log_det = forward(params, z)

    log_p = log_pz(z)
    log_q = log_q_target(x, T, a, b, kb=8.617333262145e-5)

    # log p_z(z) - log|det J| - log p_target(x)
    return jnp.mean(log_p - log_det - log_q)


# ---------------------------------------------------------------------------
# Step di training
# ---------------------------------------------------------------------------

def make_step(optimizer):
    """
    Restituisce una funzione step jit-compilata che usa l'optimizer dato.
    Passa optimizer come argomento invece di usarlo come globale.
    """
    @jax.jit
    def step(params, opt_state, key, n_samples=500, T=1.0, a=1.0, b=1.0):
        l, grads = jax.value_and_grad(loss_fn)(params, key, n_samples=n_samples, T=T, a=a, b=b)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, l

    return step


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample(params, key, n_samples=10000):
    """
    Campiona n_samples punti dalla distribuzione appresa q_flow
    """
    key, subkey = jax.random.split(key)
    z = jax.random.normal(subkey, (n_samples, params["dim"]))
    x, log_det = forward(params, z)

    return x, key 

def flow_ev_probability(params, x):
    """
    Calcola log q_flow(x) = log p_z(z) + log|det J|
    """
    z, log_det = inverse(params, x)
    log_p = log_pz(z)
    return log_p + log_det

# ---------------------------------------------------------------------------
# Reweighting
# ---------------------------------------------------------------------------
def reweight_samples(x, log_qx, T, a, b, kb=8.617333262145e-5):
    """
    Ricalcola i pesi dei campioni x ~ q_flow(x) per ottenere la distribuzione target p_target(x).
    p_target(x) ∝ exp(-E(x)/(kb*T))
    w_i = p_target(x_i) / q_flow(x_i) = exp(-E(x_i)/(kb*T)) / q_flow(x_i)
    """
    log_p_target = log_q_target(x, T, a, b, kb)
    log_weights = log_p_target - log_qx
    weights = jnp.exp(log_weights - jnp.max(log_weights))  # stabilità numerica
    weights /= jnp.sum(weights)  # normalizza
    return weights

def Ess(weights):
    """
    Calcola l'Effective Sample Size (ESS) dei pesi normalizzati.
    ESS = 1 / sum(w_i^2)
    """
    return 1.0 / jnp.sum(weights ** 2)