import jax
import jax.numpy as jnp
import optax
from functools import partial
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Configurazione statica del flow (NON differenziabile, NON un pytree di array)
# ---------------------------------------------------------------------------
# dim e d_pass sono interi Python: servono per definire shape (jax.random.normal)
# e per fare slicing (z[:, :d_pass]), operazioni che richiedono valori CONCRETI.
# Se finissero dentro lo stesso dict passato a una funzione @jax.jit non-statica,
# JAX li tratterebbe come tracer e romperebbe sia lo slicing sia le shape.
# Per questo li teniamo in un NamedTuple separato, passato come static_argnames.

class FlowStatic(NamedTuple):
    dim: int
    d_pass: tuple  # tuple di int, uno per layer (hashable -> static_argnames OK)


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
        "W3": jax.random.normal(k3, (hidden_dim, out_dim)) * 0.01,
        "b3": jnp.zeros(out_dim),
    }


def _mlp(params, x):
    h = jnp.tanh(x @ params["W1"] + params["b1"])
    h = jnp.tanh(h @ params["W2"] + params["b2"])
    return h @ params["W3"] + params["b3"]


# ---------------------------------------------------------------------------
# Coupling layer RealNVP
# ---------------------------------------------------------------------------

_SCALE_FACTOR = 2.0


def _coupling_forward(s_params, t_params, z, d_pass):
    z1 = z[:, :d_pass]
    z2 = z[:, d_pass:]

    s_raw = _mlp(s_params, z1)
    s = _SCALE_FACTOR * jnp.tanh(s_raw)
    t = _mlp(t_params, z1)

    x1 = z1
    x2 = z2 * jnp.exp(s) + t

    log_det = jnp.sum(s, axis=1)
    x = jnp.concatenate([x1, x2], axis=1)
    return x, log_det


def _coupling_inverse(s_params, t_params, x, d_pass):
    x1 = x[:, :d_pass]
    x2 = x[:, d_pass:]

    s_raw = _mlp(s_params, x1)
    s = _SCALE_FACTOR * jnp.tanh(s_raw)
    t = _mlp(t_params, x1)

    z1 = x1
    z2 = (x2 - t) * jnp.exp(-s)

    log_det = -jnp.sum(s, axis=1)
    z = jnp.concatenate([z1, z2], axis=1)
    return z, log_det


def _permute(x):
    return x[:, ::-1]


# ---------------------------------------------------------------------------
# Inizializzazione parametri
# ---------------------------------------------------------------------------

def init_params(key, dim=2, n_layers=6, hidden_dim=32):
    """
    Ritorna (params, static):
      - params: pytree SOLO di array, allenabile (lista di {"s":..., "t":...})
      - static: FlowStatic(dim, d_pass) — configurazione fissa dell'architettura
    """
    keys = jax.random.split(key, 2 * n_layers)
    layers = []
    d_pass_list = []
    for i in range(n_layers):
        d_pass_i = 1 if i % 2 == 0 else dim - 1
        d_transform_i = dim - d_pass_i

        layers.append({
            "s": _mlp_init(keys[2 * i], d_pass_i, hidden_dim, d_transform_i),
            "t": _mlp_init(keys[2 * i + 1], d_pass_i, hidden_dim, d_transform_i),
        })
        d_pass_list.append(d_pass_i)

    params = {"layers": layers}
    static = FlowStatic(dim=dim, d_pass=tuple(d_pass_list))
    return params, static


# ---------------------------------------------------------------------------
# Forward e inversa
# ---------------------------------------------------------------------------

def forward(params, static, z):
    """Mappa z ~ N(0,I) -> x ~ q_flow(x). Ritorna (x, log_det_totale)."""
    x = z
    log_det_total = jnp.zeros(z.shape[0])

    for layer, d_pass in zip(params["layers"], static.d_pass):
        x, ld = _coupling_forward(layer["s"], layer["t"], x, d_pass)
        log_det_total += ld
        x = _permute(x)

    return x, log_det_total


def inverse(params, static, x):
    """Mappa x -> z (inversa esatta). Ritorna (z, log_det_totale_inverso)."""
    z = x
    log_det_total = jnp.zeros(x.shape[0])

    for layer, d_pass in zip(reversed(params["layers"]), reversed(static.d_pass)):
        z = _permute(z)
        z, ld = _coupling_inverse(layer["s"], layer["t"], z, d_pass)
        log_det_total += ld

    return z, log_det_total


# ---------------------------------------------------------------------------
# Log-probabilità
# ---------------------------------------------------------------------------

def log_pz(z):
    d = z.shape[1]
    return -0.5 * jnp.sum(z ** 2, axis=1) - 0.5 * d * jnp.log(2.0 * jnp.pi)


def log_q_target(x, T, a=1.0, b=1.0, kb=8.617333262145e-5):
    x_slow = x[:, 0]
    x_fast = x[:, 1:]
    energy = (a / b**4) * (x_slow ** 2 - b) ** 2 + 0.5 * jnp.sum(x_fast ** 2, axis=1)
    return -energy / (kb * T)


# ---------------------------------------------------------------------------
# Loss (reverse-KL / -ELBO)
# ---------------------------------------------------------------------------

def loss_fn(params, static, key, n_samples=500, T=1.0, a=1.0, b=1.0):
    z = jax.random.normal(key, (n_samples, static.dim))
    x, log_det = forward(params, static, z)

    log_p = log_pz(z)
    log_q = log_q_target(x, T, a, b, kb=8.617333262145e-5)

    return jnp.mean(log_p - log_det - log_q)


# ---------------------------------------------------------------------------
# Step di training
# ---------------------------------------------------------------------------

def make_step(optimizer):
    @partial(jax.jit, static_argnames=("static", "n_samples"))
    def step(params, static, opt_state, key, n_samples=500, T=1.0, a=1.0, b=1.0):
        l, grads = jax.value_and_grad(loss_fn)(
            params, static, key, n_samples=n_samples, T=T, a=a, b=b
        )
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, l

    return step


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample(params, static, key, n_samples=10000):
    key, subkey = jax.random.split(key)
    z = jax.random.normal(subkey, (n_samples, static.dim))
    x, log_det = forward(params, static, z)
    return x, key


def flow_ev_probability(params, static, x):
    z, log_det = inverse(params, static, x)
    log_p = log_pz(z)
    return log_p + log_det


# ---------------------------------------------------------------------------
# Reweighting
# ---------------------------------------------------------------------------

def reweight_samples(x, log_qx, T, a, b, kb=8.617333262145e-5):
    log_p_target = log_q_target(x, T, a, b, kb)
    log_weights = log_p_target - log_qx
    weights = jnp.exp(log_weights - jnp.max(log_weights))
    weights /= jnp.sum(weights)
    return weights


def Ess(weights):
    return 1.0 / jnp.sum(weights ** 2)