import jax 
import jax.numpy as jnp
from typing import Callable

def make_config_generator(D: int, init_type: str = "normal") -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:
    def _generate_config(key: jax.Array) -> tuple[jax.Array, jax.Array]:
        if init_type == "ones":
            return jnp.ones(D), key
        elif init_type == "normal":
            key, subkey = jax.random.split(key)
            return jax.random.normal(subkey, shape=(D,)), key
        else:
            raise ValueError(f"Unknown init_type: {init_type}, the options are: 'ones', '-ones', 'normal'")
    return _generate_config

def _uniform_scalar(key: jax.Array) -> tuple[jax.Array, jax.Array]:
    key, subkey = jax.random.split(key)
    return jax.random.uniform(subkey), key
 
def make_normal_sampler(D: int) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:
    @jax.jit
    def _sample(key: jax.Array) -> tuple[jax.Array, jax.Array]:
        key, subkey = jax.random.split(key)
        return jax.random.normal(subkey, shape=(D,)), key
    return _sample
    
@jax.jit
def generate_uniform(key):
    key, subkey = jax.random.split(key)
    x = jax.random.uniform(subkey)
    return x, key

@jax.jit
def _acceptance(delta_E: jax.Array, T: jax.Array, kb: float) -> jax.Array:
    return jnp.where(delta_E <= 0, 1.0, jnp.exp(-delta_E / (T * kb)))

def make_metropolis_step(D: int, V: Callable, kb: float) -> Callable:
    sample_noise = make_normal_sampler(D)
 
    @jax.jit
    def _step(
        x:         jax.Array,
        key:       jax.Array,
        T:         jax.Array,
        step_size: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        eta, key = sample_noise(key)
        x_prop   = x + step_size * eta
        delta_E  = V(x_prop) - V(x)            # V già specializzato con (a,b)
        prob     = _acceptance(delta_E, T, kb)
        u, key   = _uniform_scalar(key)
        accept   = u < prob
        return jnp.where(accept, x_prop, x), key, accept
 
    return _step

def make_simulation(D: int, n_steps: int, V: Callable, kb: float) -> Callable:
    step_fn = make_metropolis_step(D, V, kb)
 
    @jax.jit
    def _run(
        key:       jax.Array,
        T:         jax.Array,
        step_size: jax.Array,
        initial_x: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
 
        def body(carry, _):
            x, key, acc    = carry
            x, key, accepted = step_fn(x, key, T, step_size)
            return (x, key, acc + accepted), x
 
        (x_final, key, total_acc), trajectory = jax.lax.scan(
            body, (initial_x, key, jnp.int32(0)), None, length=n_steps
        )
        return trajectory, total_acc / n_steps, key, x_final
 
    return _run