import numpy as np
from dataclasses import dataclass

@dataclass
class Trellis:
    """
    Тензорная структура графа (Trellis).
    Векторизованное представление позволяет избежать медленных Python-циклов в SISO декодере.
    """
    memory: int
    num_states: int
    next_states: np.ndarray  # shape (num_states, 2)
    outputs: np.ndarray      # shape (num_states, 2, 2) (выходные биты c1, c2)
    prev_states: np.ndarray  # shape (num_states, 2) (2 входящих состояния)
    prev_inputs: np.ndarray  # shape (num_states, 2) (2 соответствующих входных бита)

def generate_trellis() -> Trellis:
    """
    Генерирует решетку для стандартного кода (2, 1, 7)
    Полиномы: 171_8, 133_8.
    """
    memory = 6
    num_states = 64
    
    next_states = np.zeros((num_states, 2), dtype=int)
    outputs = np.zeros((num_states, 2, 2), dtype=int)
    prev_states = np.zeros((num_states, 2), dtype=int)
    prev_inputs = np.zeros((num_states, 2), dtype=int)
    
    # Вспомогательный счетчик для заполнения матриц prev_*
    prev_counter = np.zeros(num_states, dtype=int)

    for s in range(num_states):
        for u in (0, 1):
            # Состояние сдвигового регистра: u вдвигается как MSB, остальные сдвигаются
            s_next = (u << 5) | (s >> 1)
            next_states[s, u] = s_next
            
            # Извлекаем биты из состояния s для вычисления полиномов.
            # x[0] - текущий бит u, x[1] - x_{k-1}, ..., x[6] - x_{k-6}
            x = [u] + [(s >> (5 - j)) & 1 for j in range(6)]
            
            # G1 = 171_8 = 1111001_2 -> x_k, x_{k-1}, x_{k-2}, x_{k-3}, x_{k-6}
            c1 = x[0] ^ x[1] ^ x[2] ^ x[3] ^ x[6]
            # G2 = 133_8 = 1011011_2 -> x_k, x_{k-2}, x_{k-3}, x_{k-5}, x_{k-6}
            c2 = x[0] ^ x[2] ^ x[3] ^ x[5] ^ x[6]
            
            outputs[s, u, 0] = c1
            outputs[s, u, 1] = c2
            
            # Записываем обратные связи (кто пришел в s_next)
            idx = prev_counter[s_next]
            prev_states[s_next, idx] = s
            prev_inputs[s_next, idx] = u
            prev_counter[s_next] += 1
            
    return Trellis(memory, num_states, next_states, outputs, prev_states, prev_inputs)

def conv_encode(info_bits: np.ndarray) -> np.ndarray:
    """
    Кодирует массив информационных бит.
    Автоматически добавляет 6 нулевых бит (tail bits) для завершения решетки в состоянии 0.
    """
    trellis = generate_trellis()
    # Добавляем tail bits
    u_bits = np.concatenate([info_bits, np.zeros(trellis.memory, dtype=int)])
    coded_bits = np.zeros(len(u_bits) * 2, dtype=int)
    
    s = 0
    for k, u in enumerate(u_bits):
        c1, c2 = trellis.outputs[s, u]
        coded_bits[2*k] = c1
        coded_bits[2*k + 1] = c2
        s = trellis.next_states[s, u]
        
    return coded_bits