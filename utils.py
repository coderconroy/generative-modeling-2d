import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import chex
from typing import Iterator
import os

data_dir = 'dataset'
def load_dataset(dataset_name: str):
    X_train = np.load(os.path.join(data_dir, f'X_train_{dataset_name}.npy'))
    X_test = np.load(os.path.join(data_dir, f'X_test_{dataset_name}.npy'))

    return X_train, X_test

def save_samples(model_name:str, dataset_name: str, X_sample: jax.Array):
    np.save(os.path.join(data_dir, f'{model_name}_{dataset_name}_samples.npy'), X_sample)

class BatchManager(Iterator[np.ndarray]):
    def __init__(self, data: np.ndarray, batch_size: int, key: chex.PRNGKey):
        """
        Initializes the batch manager for data handling.
        
        Parameters:
            data (np.ndarray): The dataset to batch.
            batch_size (int): The size of each batch.
            key (chex.PRNGKey): The PRNG key for random operations.
        """
        self._batch_size = min(batch_size, len(data))
        self._num_batches = len(data) // self._batch_size
        self._key = hk.PRNGSequence(key)
        self._data = data
        self._reset()

    def _reset(self) -> None:
        """Resets the batch manager for a new epoch, reshuffling the data indices."""
        self._perm = np.array(jax.random.permutation(next(self._key), np.arange(len(self._data))))
        self._batch_idx = 0

    def __next__(self) -> np.ndarray:
        """Retrieves the next batch of data."""
        if self._batch_idx is None or self._batch_idx >= self._num_batches:
            raise StopIteration

        inds = self._perm[self._batch_idx * self._batch_size : (self._batch_idx + 1) * self._batch_size]
        batch = self._data[inds]
        self._batch_idx += 1
        if self._batch_idx >= self._num_batches:
            self._reset()

        return batch

    @property
    def num_batches(self) -> int:
        """Returns the total number of batches."""
        return self._num_batches