
from typing import Sequence, Callable
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

Activation = Callable[[jax.Array], jax.Array]

class Discriminator(nn.Module):
    layer_dim: Sequence[int]
    activation: Activation = nn.relu

    @nn.compact
    def __call__(self, x: jax.Array, train: bool) -> jax.Array:
        for f in self.layer_dim[:-1]:
            x = nn.Dense(f)(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=0.1, deterministic=not train)(x)
        x = nn.Dense(self.layer_dim[-1])(x)
        x = nn.sigmoid(x)
        return x

class Generator(nn.Module):
    layer_dim: Sequence[int]
    activation: Activation = nn.relu

    @nn.compact
    def __call__(self, z: jax.Array) -> jax.Array:
        for f in self.layer_dim[:-1]:
            z = nn.Dense(f)(z)
            x = nn.relu(z)
        x = nn.Dense(self.layer_dim[-1])(z)
        return x
    
def create_model(config: dict):
    discriminator = Discriminator(layer_dim=config['d_layer_dim'], activation=config['d_activation'])
    generator = Generator(layer_dim=config['g_layer_dim'], activation=config['g_activation'])
    model = {
        'dis': discriminator,
        'gen': generator
    }
    return model

def train_model(model: dict, config: dict, X_train: jax.Array, X_test: jax.Array):
    d_optimizer = optax.adam(learning_rate=config['d_learning_rate'])
    g_optimizer = optax.adam(learning_rate=config['g_learning_rate'])
    prng_seq = hk.PRNGSequence(jax.random.PRNGKey(0))
    discriminator = model['dis']
    generator = model['gen']

    @functools.partial(jax.jit, static_argnames=['train'])
    def gan_loss(d_params:chex.ArrayTree, g_params:chex.ArrayTree, batch: jax.Array, key: chex.PRNGKey, train: bool):
        # Generate sample
        key, z_key = jax.random.split(key)
        z_batch = jax.random.normal(z_key, (batch.shape[0], config['latent_dim']))
        fake_batch = generator.apply(g_params, z_batch)

        # Apply discriminator
        key1, key2 = jax.random.split(key)
        real_preds = discriminator.apply(d_params, batch, train=train, rngs={'dropout': key1})
        fake_preds = discriminator.apply(d_params, fake_batch, train=train, rngs={'dropout': key2})

        # Compute loss
        d_loss = -jnp.mean(jnp.log(real_preds) + jnp.log(1 - fake_preds))
        g_loss = -jnp.mean(jnp.log(fake_preds))
        # g_loss = jnp.mean(jnp.log(1 - fake_preds))

        return d_loss, g_loss

    @jax.jit
    def do_batch_update(batch, d_params, g_params, opt_d_state, opt_g_state, key):
        # Train discriminator
        for _ in range(config['k']):
            key, z_key = jax.random.split(key)
            compute_d_loss = lambda d_params: gan_loss(d_params, g_params, batch, z_key, train=True)[0]
            d_grad = jax.grad(compute_d_loss)(d_params)
            d_updates, opt_d_state = d_optimizer.update(d_grad, opt_d_state)
            d_params = optax.apply_updates(d_params, d_updates)

        # Train generator
        compute_g_loss = lambda g_params: gan_loss(d_params, g_params, batch, key, train=True)[1]
        g_grad = jax.grad(compute_g_loss)(g_params)
        g_updates, opt_g_state = g_optimizer.update(g_grad, opt_g_state)
        g_params = optax.apply_updates(g_params, g_updates)

        return d_params, g_params, opt_d_state, opt_g_state

    g_params = generator.init(next(prng_seq), jax.random.normal(next(prng_seq), (1, config['latent_dim'])))
    d_params = discriminator.init(next(prng_seq), X_train[:1, ...], train=False)
    opt_g_state = g_optimizer.init(g_params)
    opt_d_state = d_optimizer.init(d_params)
    bm = BatchManager(X_train, config['batch_size'], key=next(prng_seq))
    d_train_losses = []
    g_train_losses = []
    d_test_losses = []
    g_test_losses = []

    for epoch in tqdm(range(config['epochs']), "Epoch"):
        for _ in range(bm.num_batches):
            batch = next(bm)
            d_params, g_params, opt_d_state, opt_g_state = do_batch_update(
                batch, d_params, g_params, opt_d_state, opt_g_state, next(prng_seq))
        d_train_loss, g_train_loss = gan_loss(d_params, g_params, X_train, next(prng_seq), train=False)
        d_test_loss, g_test_loss = gan_loss(d_params, g_params, X_test, next(prng_seq), train=False)
        d_train_losses.append(-d_train_loss)
        g_train_losses.append(-g_train_loss)
        d_test_losses.append(-d_test_loss)
        g_test_losses.append(-g_test_loss)

    losses = {
        'd_train': d_train_losses,
        'g_train': g_train_losses,
        'd_test': d_test_losses,
        'g_test': g_test_losses
    }

    return g_params, losses

def sample_model(model: dict, g_params: chex.ArrayTree, config: dict, key: chex.PRNGKey, num_samples:int):
    z = jax.random.normal(key, (num_samples, config['latent_dim']))
    samples = model['gen'].apply(g_params, z)
    return samples

def plot_losses(losses, window_size=1):
    window = np.ones(window_size) / window_size
    d_train_losses_f = np.convolve(losses['d_train'], window, mode='valid')
    g_train_losses_f = np.convolve(losses['g_train'], window, mode='valid')
    d_test_losses_f = np.convolve(losses['d_test'], window, mode='valid')
    g_test_losses_f = np.convolve(losses['g_test'], window, mode='valid')
    x_pos = np.arange(window_size // 2, window_size // 2 + d_train_losses_f.shape[0])

    plt.plot(x_pos, d_train_losses_f, label='Discriminator Train')
    plt.plot(x_pos, g_train_losses_f, label='Generator Train')
    plt.plot(x_pos, d_test_losses_f, label='Discriminator Test')
    plt.plot(x_pos, g_test_losses_f, label='Generator Test')
    plt.xlabel('Epoch')
    plt.ylabel('Smoothed Loss')
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
