from typing import Sequence
import functools
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import flax.linen as nn
import haiku as hk
import optax
import chex
from tqdm import tqdm
from utils import BatchManager

class MLP(nn.Module):
    layer_dim: Sequence[int]
    
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for f in self.layer_dim[:-1]:
            x = nn.Dense(f)(x)
            x = nn.swish(x)
        x = nn.Dense(self.layer_dim[-1])(x)
        return x.squeeze()

def create_model(config: dict):
    model = MLP(layer_dim=config['mlp_layer_dim'])
    return model

def train_model(model: MLP, config: dict, X_train: jax.Array, X_test: jax.Array):
    optimizer = optax.adam(learning_rate=config['learning_rate'])
    prng_seq = hk.PRNGSequence(jax.random.PRNGKey(0))

    @functools.partial(jax.jit, static_argnames=("num_steps",))
    def langevin_sampling(
        params: chex.ArrayTree,
        key: chex.PRNGKey,
        step_size: float,
        initial_samples: jax.Array,
        num_steps: int,
    ) -> jax.Array:

        def scan_fn(carry, _):
            states, key = carry
            key, sk = jax.random.split(key)
            noise = jax.random.normal(sk, shape=states.shape)
            score = jax.vmap(jax.grad(lambda x: model.apply(params, x)))(states)
            next_states = states + step_size * score + jnp.sqrt(2 * step_size) * noise
            return (next_states, key), None

        states = initial_samples
        (states, _), _ = jax.lax.scan(scan_fn, (states, key), jnp.arange(num_steps))
        return states

    def ce_loss_grad(params: chex.ArrayTree, batch: jax.Array, key: chex.PRNGKey) -> float:
        # Sample from model
        key1, key2 = jax.random.split(key)
        batch_model = langevin_sampling(params, key1, 5e-3, 2 * jax.random.normal(key2, shape=batch.shape), config['sample_steps'])

        # Apply model and compute gradient for each sample
        f = lambda params, x: model.apply(params, x)
        df_data = jax.vmap(jax.grad(f), in_axes=(None, 0))(params, batch)
        df_model = jax.vmap(jax.grad(f), in_axes=(None, 0))(params, batch_model)
        grad = jax.tree.map(jnp.subtract, df_model, df_data)  # -(df_data - df_model)

        # Sum gradients across samples
        grad = jax.tree.map(lambda x: jnp.sum(x, axis=0), grad)

        return grad

    @jax.jit
    def do_batch_update(batch: jax.Array, params: chex.ArrayTree, opt_state: optax.OptState, key: chex.PRNGKey) -> tuple[float, chex.ArrayTree, optax.OptState]:
        grad = ce_loss_grad(params, batch, key)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    params = model.init(next(prng_seq), X_train[:1, ...])
    opt_state = optimizer.init(params)
    bm = BatchManager(X_train, config['batch_size'], key=next(prng_seq))
    train_energies = []
    test_energies = []
    sample_energies = []

    for epoch in tqdm(range(config['epochs']), "Epoch"):
        for _ in range(bm.num_batches):
            batch = next(bm)
            params, opt_state = do_batch_update(batch, params, opt_state, key=next(prng_seq))
        samples = langevin_sampling(params, next(prng_seq), 5e-3, 2 * jax.random.normal(next(prng_seq), shape=X_train.shape), config['sample_steps'])
        train_energy = jnp.mean(model.apply(params, X_train))
        test_energy = jnp.mean(model.apply(params, X_test))
        sample_energy = jnp.mean(model.apply(params, samples))
        train_energies.append(train_energy)
        test_energies.append(test_energy)
        sample_energies.append(sample_energy)

    energies = {
        'train': train_energies,
        'test': test_energies,
        'sample': sample_energies
    }

    return params, energies

def sample_model(model: MLP, params: chex.ArrayTree, key: chex.PRNGKey, num_samples: int):

    @functools.partial(jax.jit, static_argnames=("num_steps",))
    def langevin_sampling(
        params: chex.ArrayTree,
        key: chex.PRNGKey,
        step_size: float,
        initial_samples: jax.Array,
        num_steps: int,
    ) -> jax.Array:

        def scan_fn(carry, _):
            states, key = carry
            key, sk = jax.random.split(key)
            noise = jax.random.normal(sk, shape=states.shape)
            score = jax.vmap(jax.grad(lambda x: model.apply(params, x)))(states)
            next_states = states + step_size * score + jnp.sqrt(2 * step_size) * noise
            return (next_states, key), None

        states = initial_samples
        (states, _), _ = jax.lax.scan(scan_fn, (states, key), jnp.arange(num_steps))
        return states
        
    key1, key2 = jax.random.split(key)
    samples = langevin_sampling(params, key1, 5e-3, 2 * jax.random.normal(key2, shape=(num_samples, 2)), 1000)

    return samples

def plot_energies(energies, window_size=1):
    window = np.ones(window_size) / window_size
    train_energies_f = np.convolve(energies['train'], window, mode='valid')
    test_energies_f = np.convolve(energies['test'], window, mode='valid')
    sample_energies_f = np.convolve(energies['sample'], window, mode='valid')
    x_pos = np.arange(window_size // 2, window_size // 2 + train_energies_f.shape[0])

    plt.plot(x_pos, train_energies_f, label='Train')
    plt.plot(x_pos, test_energies_f, label='Test')
    plt.plot(x_pos, sample_energies_f, label='Sample')
    plt.xlabel('Epoch')
    plt.ylabel('Energy Function')
    plt.legend()
    plt.show()

def plot_data(X_train: jax.Array, samples: jax.Array):
    plt.scatter(samples[:, 0], samples[:, 1], marker='.', label='Sampled')
    plt.scatter(X_train[:, 0], X_train[:, 1], alpha=0.2, marker='o', label='Train')
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.axis('equal')
    plt.legend()
    plt.show()