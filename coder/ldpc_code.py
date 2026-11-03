import numpy as np

def make_systematic(H: np.ndarray) -> np.ndarray:
    """
    Приводит проверочную матрицу H к систематическому виду H_sys = [P^T | I] 
    методом Гаусса над GF(2) и возвращает порождающую матрицу G = [I | P].
    
    Алгоритм обрабатывает столбцы справа налево, формируя единичную матрицу в 
    правой части. Линейно зависимые строки зануляются и отбрасываются.
    
    :param H: Проверочная матрица LDPC кода.
    :return: Порождающая матрица G.
    :raises ValueError: Если правые столбцы матрицы линейно зависимы и 
                        приведение к [P^T | I] невозможно без перестановок столбцов.
    """
    m, n = H.shape
    H_sys = H.copy()
    
    # Индекс текущей строки для поиска ведущего элемента (пивота)
    r = m - 1
    
    # Проходим по столбцам справа налево
    for c in range(n - 1, -1, -1):
        if r < 0:
            break
            
        # Ищем единицу в текущем столбце среди строк от 0 до r включительно
        pivot = np.argmax(H_sys[:r+1, c])
        
        if H_sys[pivot, c] == 1:
            # Если пивот не на месте r, меняем строки местами
            if pivot != r:
                H_sys[[r, pivot]] = H_sys[[pivot, r]]
                
            # Зануляем остальные единицы в этом столбце (операция XOR)
            for i in range(m):
                if i != r and H_sys[i, c] == 1:
                    H_sys[i] = H_sys[i] ^ H_sys[r]
            r -= 1
            
    # r указывает на последнюю зануленную строку.
    # Количество независимых строк (ранг) равно m - (r + 1).
    rank = m - (r + 1)
    
    # Оставляем только линейно независимые строки (они опустились вниз)
    H_sys = H_sys[r+1:, :]
    
    # Строгая проверка: убеждаемся, что справа образовалась матрица I
    if rank == 0 or not np.array_equal(H_sys[:, n-rank:], np.eye(rank, dtype=int)):
        raise ValueError(
            "Матрица не может быть приведена к виду [P^T | I] без перестановок "
            "столбцов (правая часть линейно зависима)."
        )
        
    # Извлекаем P^T и формируем G
    k = n - rank
    P_T = H_sys[:, :k]
    P = P_T.T
    
    G = np.hstack((np.eye(k, dtype=int), P))
    return G

def generate_regular_h_matrix(n_cols: int, d_v: int, d_c: int) -> np.ndarray:
    """
    Генерирует проверочную матрицу H для регулярного LDPC кода (Gallager).
    Гарантирует, что матрица приводима к систематическому виду [P^T | I].
    
    :param n_cols: Длина кодового слова (N).
    :param d_v: Вес столбца (число единиц в столбце, Variable Node degree).
    :param d_c: Вес строки (число единиц в строке, Check Node degree).
    :return: Проверочная матрица H.
    """
    if d_v >= d_c:
        raise ValueError("Для LDPC кода d_v должно быть строго меньше d_c.")
    if n_cols % d_c != 0:
        raise ValueError(f"Длина кодового слова {n_cols} должна быть кратна d_c={d_c}.")
        
    n_rows = (n_cols * d_v) // d_c
    block_rows = n_rows // d_v
    
    # Создаем базовый субблок, где единицы идут лесенкой
    H_base = np.zeros((block_rows, n_cols), dtype=int)
    for i in range(block_rows):
        H_base[i, i * d_c : (i + 1) * d_c] = 1
        
    max_attempts = 1000
    for _ in range(max_attempts):
        H = np.zeros((n_rows, n_cols), dtype=int)
        
        # Для максимальной рандомизации переставляем столбцы во всех d_v субблоках
        for v in range(d_v):
            p = np.random.permutation(n_cols)
            H[v * block_rows : (v + 1) * block_rows] = H_base[:, p]
            
        # Проверяем матрицу на возможность приведения к систематическому виду
        try:
            make_systematic(H)
            return H  # Успех, возвращаем матрицу
        except ValueError:
            continue  # Правая часть оказалась вырожденной, пробуем снова
            
    raise RuntimeError(
        "Не удалось сгенерировать матрицу H, приводимую к [P^T | I], "
        f"за {max_attempts} попыток. Попробуйте изменить параметры."
    )

def ldpc_encode(u: np.ndarray, G: np.ndarray) -> np.ndarray:
    """
    Кодирует вектор информационных бит u с помощью порождающей матрицы G.
    
    :param u: Вектор информационных бит (1D массив).
    :param G: Порождающая матрица в систематическом виде.
    :return: Закодированное слово (1D массив).
    """
    return (u @ G) % 2