import jax
import jax.numpy as jnp
from typing import Callable
from functools import partial
import boltzmann_gen as bg


def make_flow_metropolis_step(V: Callable, kb: float) -> Callable:
    """
    Singolo step della catena Flow-MCMC.
    La proposal è un campione indipendente dal flow (non dipende da x corrente).
    Il bilancio dettagliato richiede il rapporto q(x)/q(x') nella probabilità di accettazione:

        A(x -> x') = min(1, p(x') * q(x) / (p(x) * q(x')))

    dove p è la target e q è la distribuzione del flow.
    """
    @partial(jax.jit, static_argnames=("static",))
    def _step(
        x:         jax.Array,        # config corrente, shape (D,)
        log_qx:    jax.Array,        # log q(x) corrente, scalar
        key:       jax.Array,
        T:         jax.Array,
        params:    dict,             # parametri allenabili del flow (array)
        static:    bg.FlowStatic,    # dim, d_pass: configurazione statica
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:

        # --- proposal dal flow ---
        key, subkey = jax.random.split(key)
        z_prop = jax.random.normal(subkey, shape=(1, static.dim))
        x_prop, log_det_fwd = bg.forward(params, static, z_prop)
        x_prop = x_prop[0]  # (D,)

        # --- log q(x') ---
        log_qx_prop = bg.log_pz(z_prop)[0] - log_det_fwd[0]

        # --- log p(x) e log p(x') ---
        log_px      = -V(x)       / (kb * T)
        log_px_prop = -V(x_prop)  / (kb * T)

        # --- log acceptance ratio ---
        log_accept = log_px_prop + log_qx - log_px - log_qx_prop
        log_accept = jnp.minimum(0.0, log_accept)

        # --- accettazione ---
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey)
        accept = jnp.log(u) < log_accept

        x_new      = jnp.where(accept, x_prop,    x)
        log_qx_new = jnp.where(accept, log_qx_prop, log_qx)

        return x_new, log_qx_new, key, accept

    return _step


def make_flow_simulation_batched(n_steps, V, kb):
    @partial(jax.jit, static_argnames=("static",))
    def _run(key, T, initial_x, params, static):
        key, subkey = jax.random.split(key)
        z_all = jax.random.normal(subkey, (n_steps, static.dim))
        x_prop_all, log_det_all = bg.forward(params, static, z_all)   # UN SOLO forward batched
        log_qprop_all = bg.log_pz(z_all) - log_det_all
        log_pprop_all = -jax.vmap(V)(x_prop_all) / (kb * T)

        log_qx_init = bg.flow_ev_probability(params, static, initial_x[None, :])[0]

        def body(carry, inp):
            x, log_qx, key, acc = carry
            x_prop, log_qprop, log_pprop = inp
            log_px = -V(x) / (kb * T)
            log_accept = jnp.minimum(0.0, log_pprop + log_qx - log_px - log_qprop)
            key, subkey = jax.random.split(key)
            accept = jnp.log(jax.random.uniform(subkey)) < log_accept
            x_new = jnp.where(accept, x_prop, x)
            log_qx_new = jnp.where(accept, log_qprop, log_qx)
            return (x_new, log_qx_new, key, acc + accept), x_new

        (x_final, _, key, total_acc), trajectory = jax.lax.scan(
            body, (initial_x, log_qx_init, key, jnp.int32(0)),
            (x_prop_all, log_qprop_all, log_pprop_all)
        )
        return trajectory, total_acc / n_steps, key, x_final
    return _run