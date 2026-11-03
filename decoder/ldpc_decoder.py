import numpy as np

def _gf2_rank(H: np.ndarray) -> int:
    """
    Быстрое вычисление ранга матрицы над полем Галуа GF(2).
    Используется для точного определения количества информационных бит (K = N - rank).
    """
    A = np.array(H, dtype=int)
    m, n = A.shape
    r = 0
    for c in range(n):
        if r >= m:
            break
        pivot = np.argmax(A[r:, c]) + r
        if A[pivot, c]:
            if pivot != r:
                A[[r, pivot]] = A[[pivot, r]]
            # Векторизованный XOR для оставшейся части столбца
            idx = np.where(A[r+1:, c])[0] + r + 1
            A[idx] ^= A[r]
            r += 1
    return r

def ldpc_siso_decode(apriori_llr: np.ndarray, H: np.ndarray, max_iter: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """
    Soft-In Soft-Out декодер для регулярного LDPC кода (Log-SPA).
    Векторизованная реализация с использованием списков смежности (edge mapping).
    
    :param apriori_llr: Априорные LLR кодовых бит (1D массив).
    :param H: Проверочная матрица LDPC.
    :param max_iter: Максимальное количество итераций BP.
    :return: (llr_c_ext, llr_u_app)
             llr_c_ext - массив Extrinsic LLR всех кодовых бит (размер N).
             llr_u_app - массив апостериорных LLR информационных бит (размер K).
    """
    M, N = H.shape
    
    # Списки смежности (adjacency lists) в виде индексов неквалифицированных ребер
    rows, cols = np.nonzero(H)
    E = len(rows)
    
    d_c = np.sum(H[0])
    d_v = np.sum(H[:, 0])
    
    if E != M * d_c or E != N * d_v:
        raise ValueError("Данный декодер оптимизирован только для строго регулярных LDPC кодов.")
        
    # Индексы для перестановки массива ребер из порядка Check Node (по умолчанию в np.nonzero) 
    # в порядок Variable Node
    vn_sort_idx = np.argsort(cols, kind='stable')
    inv_vn_sort_idx = np.empty_like(vn_sort_idx)
    inv_vn_sort_idx[vn_sort_idx] = np.arange(E)
    
    # Сообщения V2C инициализируются априорными LLR. 
    # Поскольку edges отсортированы по строкам (CN), берем значения по индексам столбцов (cols).
    msg_v2c = apriori_llr[cols].astype(float)
    total_sum = apriori_llr.copy()
    
    for _ in range(max_iter):
        # --- 1. Обновление проверочных узлов (Check Nodes) ---
        # Формируем матрицу сообщений для каждого CN (размер M x d_c)
        msg_v2c_resh = msg_v2c.reshape(M, d_c)
        
        t = np.tanh(msg_v2c_resh / 2.0)
        t = np.clip(t, -1.0 + 1e-15, 1.0 - 1e-15)
        
        # Надежная защита от деления на ноль и потери знака при underflow
        t = np.where((t >= 0) & (t < 1e-15), 1e-15, t)
        t = np.where((t < 0) & (t > -1e-15), -1e-15, t)
        
        prod_t = np.prod(t, axis=1, keepdims=True)
        
        # Исключающее произведение для каждого ребра
        val = np.clip(prod_t / t, -1.0 + 1e-15, 1.0 - 1e-15)
        msg_c2v_resh = 2.0 * np.arctanh(val)
        
        # Переводим обратно в плоский массив (порядок CN)
        msg_c2v = msg_c2v_resh.ravel()
        
        # --- 2. Обновление переменных узлов (Variable Nodes) ---
        # Перегруппировываем сообщения в порядок VN и делаем reshape (N x d_v)
        msg_c2v_by_vn = msg_c2v[vn_sort_idx].reshape(N, d_v)
        
        # Апостериорный LLR: априорный + сумма всех входящих от CN
        total_sum = apriori_llr + np.sum(msg_c2v_by_vn, axis=1)
        
        # --- 3. Ранняя остановка (Syndrome Check) ---
        # В BPSK 1 -> +1 (LLR > 0), 0 -> -1 (LLR < 0).
        u_hat = (total_sum > 0).astype(int)
        if not np.any((H @ u_hat) % 2):
            break  # Синдром нулевой, все проверки сошлись!
            
        # Экстринсивная информация для следующей итерации (разность)
        msg_v2c_by_vn = total_sum[:, np.newaxis] - msg_c2v_by_vn
        
        # Возвращаем сообщения в порядок CN для следующего цикла
        msg_v2c = msg_v2c_by_vn.ravel()[inv_vn_sort_idx]
        
    # --- Финальные вычисления ---
    llr_posteriori = total_sum
    llr_c_ext = llr_posteriori - apriori_llr
    
    # Так как матрица приводилась к систематическому виду G = [I | P],
    # первые K бит являются информационными.
    K = N - _gf2_rank(H)
    llr_u_app = llr_posteriori[:K]
    
    return llr_c_ext, llr_u_app