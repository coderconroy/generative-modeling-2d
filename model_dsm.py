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
        return x

def create_model(config: dict):
    model = MLP(layer_dim=config['mlp_layer_dim'])
    return model

def train_model(model: MLP, config: dict, X_train: jax.Array, X_test: jax.Array):
    optimizer = optax.adam(learning_rate=config['learning_rate'])
    prng_seq = hk.PRNGSequence(jax.random.PRNGKey(0))

    def dsm_loss(params: chex.ArrayTree, batch: jax.Array, key: chex.PRNGKey, std: float, k: int) -> float:
        n, m = batch.shape
        batch = jnp.tile(batch[None, :, :], (k, 1, 1))
        noise = jax.random.normal(key, batch.shape) * std
        noised_batch = noise + batch
        fs = model.apply(params, noised_batch.reshape(k * n, m)).reshape(k, n, m)
        loss = jnp.sum(jnp.square(noised_batch + std**2 * fs - batch)) / k
        return loss

    @jax.jit
    def do_batch_update(batch: jax.Array, params: chex.ArrayTree, opt_state: optax.OptState, key: chex.PRNGKey) -> tuple[float, chex.ArrayTree, optax.OptState]:
        loss, grad = jax.value_and_grad(dsm_loss)(params, batch, key, std=0.1, k=100)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state

    params = model.init(next(prng_seq), X_train[:1, ...])
    opt_state = optimizer.init(params)
    bm = BatchManager(X_train, config['batch_size'], key=next(prng_seq))
    train_losses = []
    test_losses = []

    for epoch in tqdm(range(config['epochs']), "Epoch"):
        batch_loss = 0
        for _ in range(bm.num_batches):
            batch = next(bm)
            train_loss, params, opt_state = do_batch_update(batch, params, opt_state, key=next(prng_seq))
            batch_loss += train_loss
        test_loss = dsm_loss(params, X_test, key=next(prng_seq), std=config['std'], k=config['k'])
        train_losses.append(batch_loss / X_train.shape[0])
        test_losses.append(test_loss / X_test.shape[0])

    losses = {
        'train': train_losses,
        'test': test_losses
    }

    return params, losses

def sample_model(model: MLP, params: chex.ArrayTree, key: chex.PRNGKey, num_samples: int):

    @functools.partial(jax.jit, static_argnames=("num_steps",))
    def langevin_sampling(params: chex.ArrayTree, key: chex.PRNGKey, step_size: float, initial_samples: jax.Array, num_steps: int) -> jax.Array:
        def scan_fn(carry, _):
            states, key = carry
            key, sk = jax.random.split(key)
            noise = jax.random.normal(sk, shape=states.shape)
            next_states = states + step_size * model.apply(params, states) + jnp.sqrt(2 * step_size) * noise
            return (next_states, key), None

        states = initial_samples
        (states, _), _ = jax.lax.scan(scan_fn, (states, key), jnp.arange(num_steps))
        return states

    key1, key2 = jax.random.split(key)
    samples = langevin_sampling(params, key1, 5e-3, 2 * jax.random.normal(key2, shape=(num_samples, 2)), 1000)

    return samples

def plot_losses(losses):
    plt.plot(losses['train'], label='Train')
    plt.plot(losses['test'], label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Denoising Score Matching Loss')
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
