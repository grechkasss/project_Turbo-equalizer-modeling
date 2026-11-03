import numpy as np

class RandomInterleaver:
    """
    Псевдослучайный перемежитель.
    Обеспечивает строгое воспроизведение паттерна перемежения через seed.
    """
    def __init__(self, size: int, seed: int = 42):
        self.size = size
        rng = np.random.default_rng(seed)
        
        # Прямая перестановка
        self.p = rng.permutation(size)
        # Обратная перестановка (O(N) построение)
        self.inv_p = np.empty(size, dtype=int)
        self.inv_p[self.p] = np.arange(size)
        
    def interleave(self, data: np.ndarray) -> np.ndarray:
        """Перемешивает массив."""
        if len(data) != self.size:
            raise ValueError(f"Размер данных {len(data)} не совпадает с размером перемежителя {self.size}")
        return data[self.p]
        
    def deinterleave(self, data: np.ndarray) -> np.ndarray:
        """Восстанавливает исходный порядок массива."""
        if len(data) != self.size:
            raise ValueError(f"Размер данных {len(data)} не совпадает с размером перемежителя {self.size}")
        return data[self.inv_p]