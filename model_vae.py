from typing import Sequence
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import flax.linen as nn
import haiku as hk
import optax
import chex
from tqdm import tqdm
from utils import BatchManager

class Encoder(nn.Module):
    layer_dim: Sequence[int]
    latent_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for f in self.layer_dim:
            x = nn.Dense(f)(x)
            x = nn.swish(x)
        x = nn.Dense(self.latent_dim * 2)(x)
        mean = x[..., :self.latent_dim]
        log_var = x[..., self.latent_dim:]
        return mean, log_var

class Decoder(nn.Module):
    layer_dim: Sequence[int]
    output_dim: int

    @nn.compact
    def __call__(self, z: jax.Array) -> jax.Array:
        for f in self.layer_dim:
            z = nn.Dense(f)(z)
            z = nn.swish(z)
        x_recon = nn.Dense(self.output_dim)(z)
        return x_recon

class VAE(nn.Module):
    enc_layer_dim: Sequence[int]
    dec_layer_dim: Sequence[int]
    latent_dim: int
    output_dim: int

    def setup(self):
        self.encoder = Encoder(self.enc_layer_dim, self.latent_dim)
        self.decoder = Decoder(self.dec_layer_dim, self.output_dim)

    def __call__(self, x: jax.numpy.ndarray, key: chex.PRNGKey):
        mean, log_var = self.encoder(x)
        z = mean + jnp.exp(0.5 * log_var) * jax.random.normal(key, mean.shape)
        x_recon = self.decoder(z)
        return x_recon, mean, log_var
    
def create_model(config: dict):
    model = VAE(
        enc_layer_dim=config['enc_layer_dim'],
        dec_layer_dim=config['dec_layer_dim'],
        latent_dim=config['latent_dim'],
        output_dim=config['output_dim']
    )
    return model

def train_model(model: VAE, config: dict, X_train: jax.Array, X_test: jax.Array):
    optimizer = optax.adam(learning_rate=config['learning_rate'])
    prng_seq = hk.PRNGSequence(jax.random.PRNGKey(0))

    def vae_loss(params: chex.ArrayTree, batch: jax.Array, key: chex.PRNGKey):
        batch_recon, mean, log_var = model.apply(params, batch, key)
        recon_loss = jnp.mean(jnp.square(batch - batch_recon))  # Reconstruction loss
        kl_div = - jnp.mean( jnp.sum(1 + log_var - jnp.square(mean) - jnp.exp(log_var), axis=1))  # KL divergence
        return recon_loss + config['beta'] * kl_div

    @jax.jit
    def do_batch_update(batch: jax.Array, params: chex.ArrayTree, opt_state: optax.OptState, key: chex.PRNGKey) -> tuple[float, chex.ArrayTree, optax.OptState]:
        loss, grad = jax.value_and_grad(vae_loss)(params, batch, key)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state

    params = model.init(next(prng_seq), X_train, next(prng_seq))
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
        test_loss = vae_loss(params, X_test, next(prng_seq))
        train_losses.append(batch_loss / bm.num_batches)
        test_losses.append(test_loss)

    losses = {
        'train': train_losses,
        'test': test_losses
    }

    return params, losses

def sample_model(model: VAE, params: chex.ArrayTree, key: chex.PRNGKey, num_samples: int, X_train: jax.Array):
    x, _, _ = model.apply(params, X_train, key)
    return x

def plot_losses(losses):
    plt.plot(losses['train'][1:], label='Train')
    plt.plot(losses['test'][1:], label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Empirical ELBO')
    plt.legend()
    plt.show()

def plot_data(X_train: jax.Array, samples: jax.Array):
    plt.scatter(samples[:, 0], samples[:, 1], marker='.', label='Model')
    plt.scatter(X_train[:, 0], X_train[:, 1], alpha=0.2, marker='o', label='True')
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.axis('equal')
    plt.legend()
    plt.show() 
