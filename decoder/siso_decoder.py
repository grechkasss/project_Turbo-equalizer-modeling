import numpy as np
from scipy.special import logsumexp
from conv_code import Trellis

def siso_decode(apriori_llr_c: np.ndarray, trellis: Trellis, max_log_map: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Soft-In Soft-Out декодер на базе алгоритма BCJR.
    
    :param apriori_llr_c: Априорные LLR кодовых бит (1D массив от эквалайзера).
    :param trellis: Структура графа свёрточного кода.
    :param max_log_map: Если True, использует упрощенный Max-Log-MAP. Иначе — точный Log-MAP (logsumexp).
    :return: (llr_c_ext, llr_u_app)
             llr_c_ext - массив Extrinsic LLR кодовых бит для петли обратной связи (shape = input).
             llr_u_app - массив апостериорных LLR информационных бит (без tail bits).
    """
    if len(apriori_llr_c) % 2 != 0:
        raise ValueError("Длина apriori_llr_c должна быть четной (rate 1/2).")
    
    T = len(apriori_llr_c) // 2
    N_info = T - trellis.memory
    
    # Reshape для удобной работы [Время, 2 бита]
    L_a = apriori_llr_c.reshape(T, 2)
    
    # 1. Быстрый расчет метрик ветвей (Gamma)
    # Формула: gamma_k(s, u) = c1 * L_a[k, 0] + c2 * L_a[k, 1]
    # Используем np.einsum для мгновенного тензорного умножения без циклов.
    # shape gamma: (T, 64, 2) -> (время, состояние, входной бит)
    gamma = np.einsum('suj,kj->ksu', trellis.outputs, L_a)
    
    # 2. Прямая рекурсия (Alpha)
    alpha = np.full((T + 1, trellis.num_states), -np.inf)
    alpha[0, 0] = 0.0  # Начинаем строго из нулевого состояния
    
    s_prev = trellis.prev_states  # shape (64, 2)
    u_prev = trellis.prev_inputs  # shape (64, 2)
    
    for k in range(T):
        # metric shape: (64, 2). Извлекаем метрики приходящих ветвей для всех 64 состояний разом!
        metric = alpha[k, s_prev] + gamma[k, s_prev, u_prev]
        
        if max_log_map:
            alpha[k+1] = np.max(metric, axis=1)
        else:
            alpha[k+1] = logsumexp(metric, axis=1)
            
        # Нормализация для предотвращения переполнения float64
        alpha[k+1] -= np.max(alpha[k+1])
        
    # 3. Обратная рекурсия (Beta)
    beta = np.full((T + 1, trellis.num_states), -np.inf)
    # Благодаря tail bits, мы 100% знаем, что решетка завершилась в состоянии 0
    beta[T, 0] = 0.0 
    
    s_next = trellis.next_states  # shape (64, 2)
    
    for k in range(T - 1, -1, -1):
        metric = beta[k+1, s_next] + gamma[k]
        
        if max_log_map:
            beta[k] = np.max(metric, axis=1)
        else:
            beta[k] = logsumexp(metric, axis=1)
            
        beta[k] -= np.max(beta[k])
        
    # 4. Совместная вероятность переходов (Sigma) и расчет LLR
    # sigma shape: (T, 64, 2)
    sigma = alpha[:-1, :, np.newaxis] + gamma + beta[1:, s_next]
    
    # --- APP LLR для информационных бит ---
    if max_log_map:
        llr_u_full = np.max(sigma[:, :, 1], axis=1) - np.max(sigma[:, :, 0], axis=1)
    else:
        llr_u_full = logsumexp(sigma[:, :, 1], axis=1) - logsumexp(sigma[:, :, 0], axis=1)
        
    # Отбрасываем LLR для tail bits, они нам не нужны
    llr_u_app = llr_u_full[:N_info]
    
    # --- APP и Extrinsic LLR для кодовых бит ---
    llr_c_ext = np.zeros_like(L_a)
    
    for j in range(2):
        # Маскируем невалидные переходы значением -np.inf
        mask_1 = (trellis.outputs[:, :, j] == 1)
        mask_0 = ~mask_1
        
        sigma_1 = np.where(mask_1, sigma, -np.inf)
        sigma_0 = np.where(mask_0, sigma, -np.inf)
        
        if max_log_map:
            app_1 = np.max(sigma_1.reshape(T, -1), axis=1)
            app_0 = np.max(sigma_0.reshape(T, -1), axis=1)
        else:
            app_1 = logsumexp(sigma_1.reshape(T, -1), axis=1)
            app_0 = logsumexp(sigma_0.reshape(T, -1), axis=1)
            
        # Extrinsic = A posteriori - A priori
        llr_c_ext[:, j] = (app_1 - app_0) - L_a[:, j]
        
    return llr_c_ext.ravel(), llr_u_app