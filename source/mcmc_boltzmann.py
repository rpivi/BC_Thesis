import jax
import jax.numpy as jnp
from typing import Callable
import boltzmann_gen as bg


def make_flow_metropolis_step(V: Callable, kb: float) -> Callable:
    """
    Singolo step della catena Flow-MCMC.
    La proposal è un campione indipendente dal flow (non dipende da x corrente).
    Il bilancio dettagliato richiede il rapporto q(x)/q(x') nella probabilità di accettazione:

        A(x -> x') = min(1, p(x') * q(x) / (p(x) * q(x')))

    dove p è la target e q è la distribuzione del flow.
    """
    @jax.jit
    def _step(
        x:         jax.Array,   # config corrente, shape (D,)
        log_qx:    jax.Array,   # log q(x) corrente, scalar
        key:       jax.Array,
        T:         jax.Array,
        bg_params: dict,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:

        # --- proposal dal flow ---
        key, subkey = jax.random.split(key)
        z_prop = jax.random.normal(subkey, shape=(1, bg_params["dim"]))
        x_prop, _ = bg.forward(bg_params, z_prop)
        x_prop = x_prop[0]  # (D,)

        # --- log q(x') ---
        log_qx_prop = bg.flow_ev_probability(bg_params, x_prop[None, :])[0]

        # --- log p(x) e log p(x') ---
        log_px      = -V(x)       / (kb * T)
        log_px_prop = -V(x_prop)  / (kb * T)

        # --- log acceptance ratio ---
        # log A = log p(x') + log q(x) - log p(x) - log q(x')
        log_accept = log_px_prop + log_qx - log_px - log_qx_prop
        log_accept = jnp.minimum(0.0, log_accept)  # min(0, log A)

        # --- accettazione ---
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey)
        accept = jnp.log(u) < log_accept

        x_new      = jnp.where(accept, x_prop,    x)
        log_qx_new = jnp.where(accept, log_qx_prop, log_qx)

        return x_new, log_qx_new, key, accept

    return _step


def make_flow_simulation(n_steps: int, V: Callable, kb: float) -> Callable:
    """
    Esegue n_steps di Flow-MCMC.
    Ritorna (trajectory, acceptance_rate, key, x_final) — stessa interfaccia di make_simulation in mcmc.py.
    """
    step_fn = make_flow_metropolis_step(V, kb)

    @jax.jit
    def _run(
        key:       jax.Array,
        T:         jax.Array,
        initial_x: jax.Array,   # shape (D,)
        bg_params: dict,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:

        # log q della configurazione iniziale
        log_qx_init = bg.flow_ev_probability(bg_params, initial_x[None, :])[0]

        def body(carry, _):
            x, log_qx, key, acc = carry
            x, log_qx, key, accepted = step_fn(x, log_qx, key, T, bg_params)
            return (x, log_qx, key, acc + accepted), x

        (x_final, _, key, total_acc), trajectory = jax.lax.scan(
            body, (initial_x, log_qx_init, key, jnp.int32(0)), None, length=n_steps
        )
        return trajectory, total_acc / n_steps, key, x_final

    return _run
