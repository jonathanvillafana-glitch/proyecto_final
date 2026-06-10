import numpy as np

from modelo import Stroke, render
from metricas import perdida_compuesta


# ─────────────────────────────────────────
# BOUNDS (restricciones de cada parámetro)
# ─────────────────────────────────────────
# (mínimo, máximo) para cada parámetro del vector

# El vector tiene 9 posiciones
# mismo orden que Stroke.to_vector()
# [x, y, theta, r1, r2, R, G, B, alpha]

BOUNDS = [
    (0.0,    1.0),    # x — posición horizontal, 0=izq, 1=der
    (0.0,    1.0),    # y — posición vertical, 0=arr, 1=aba
    (0.0, 2 * np.pi), # theta — rotación en radianes (0 a 2π = 6.2832)
    (0.01,   0.30),   # r1 — semieje principal, mínimo 0.01 para ser visible
    (0.01,   0.30),   # r2 — semieje secundario
    (0.0,    1.0),    # R  — canal rojo
    (0.0,    1.0),    # G — canal verde
    (0.0,    1.0),    # B — canal azul
    (0.1,    0.9),    # alpha — opacidad, evitamos 0 (invisible) y 1 (tapa todo)
]

# ────────────────────────────
# PESOS DE LA FUNCIÓN OBJETIVO
# ────────────────────────────
# L = 0.6·MSE + 0.35·(1-SSIM) + 0.0005·N_strokes
# Los definimos como constantes aquí

ALPHA_LOSS = 0.6
BETA_LOSS  = 0.35
GAMMA_LOSS = 0.0005
SHAPE = 'ellipse'

# ───────────────
# _evaluar_stroke
# ───────────────
# función interna

def _evaluar_stroke(vector, canvas, imagen_objetivo, n_strokes_actuales):
    """
    Con un vector de 9 números construimos el stroke, lo dibujamos sobre
    una copia del canvas y devolvemos la pérdida resultante

    El optimizador la llama cientos de veces variando 'vector' en cada llamada
    para encontrar el mínimo

    Necesitamos EVALUAR el stroke sin modificar el canvas real.
    Si lo modificáramos, cada evaluación lo alteraría permanentemente
    y el optimizador perdería el hilo completamente.

    Args:
        vector             : array de 9 floats [x,y,theta,r1,r2,R,G,B,alpha]
        canvas             : numpy array (H,W,3) — estado actual del canvas
        imagen_objetivo    : numpy array (H,W,3) — la imagen que queremos copiar
        n_strokes_actuales : int — cuántos strokes hay ya en el canvas

    Returns:
        float — valor de pérdida. Menor = mejor
    """
    # Paso 1: reconstruir el Stroke desde el vector de números
    # from_vector es el inverso de to_vector, definido en modelo.py
    stroke = Stroke.from_vector(vector, shape=SHAPE)

    # Paso 2: obtener dimensiones del canvas
    H, W, _ = canvas.shape

    # Paso 3: renderizar el stroke sobre una copia del canvas
    # render() recibe (lista_strokes, H, W, canvas_base=...)
    canvas_evaluacion = render(
        [stroke],           # lista con un solo stroke nuevo
        H, W,               # dimensiones
        canvas_base=canvas  # render() hace .copy() internamente, no modifica canvas
    )

    # Paso 4: calcular la pérdida con los pesos del proyecto
    # Pasamos n_strokes_actuales + 1 porque ya incluimos este stroke nuevo
    perdida = perdida_compuesta(
        imagen_objetivo,
        canvas_evaluacion,
        n_strokes_actuales + 1,
        a=ALPHA_LOSS,
        b=BETA_LOSS,
        g=GAMMA_LOSS
    )

    return perdida


# ───────────────────────────────────────────────────────
# _gradiente_numerico  (diferencias finitas centradas)
# ───────────────────────────────────────────────────────

def _gradiente_numerico(vector, canvas, imagen_objetivo, n_strokes_actuales, eps=1e-5):
    """
    Aproximamos el gradiente de _evaluar_stroke respecto a los 9 parámetros
    usando diferencias finitas centradas de orden O(eps²)

    Para cada parámetro i lo perturbamos hacia adelante y hacia atrás,
    manteniendo los demás fijos:

        ∂L/∂vᵢ ≈ [ L(v + eps·eᵢ) - L(v - eps·eᵢ) ] / (2·eps)

    donde eᵢ es el vector canónico con 1 en la posición i.

    Costo: 2 × 9 = 18 evaluaciones de _evaluar_stroke por llamada

    ¿Por qué centradas y no hacia adelante (L(v+eps) - L(v)) / eps?
    Las centradas tienen error de truncamiento O(eps²) vs O(eps) de las
    hacia adelante. Con eps=1e-5 la diferencia es ~1e-10 vs ~1e-5:
    cinco órdenes de magnitud más preciso al mismo costo computacional.

    Args:
        vector             : array de 9 floats — punto actual en el espacio
        canvas             : numpy array (H,W,3) — canvas actual (no se modifica)
        imagen_objetivo    : numpy array (H,W,3) — imagen original
        n_strokes_actuales : int — strokes ya en el canvas
        eps                : float — tamaño de perturbación (default 1e-5)

    Returns:
        grad : numpy array (9,) — gradiente aproximado en el punto 'vector'
    """
    vector = np.array(vector, dtype=np.float64)
    n      = len(vector)          # 9 parámetros
    grad   = np.zeros(n)

    for i in range(n):
        # Vector canónico eᵢ: solo la posición i vale 1
        ei = np.zeros(n)
        ei[i] = 1.0

        # Perturbación hacia adelante: v + eps·eᵢ
        v_adelante = vector + eps * ei
        # Perturbación hacia atrás:   v - eps·eᵢ
        v_atras    = vector - eps * ei

        # Evaluamos L en ambos extremos
        L_adelante = _evaluar_stroke(v_adelante, canvas, imagen_objetivo, n_strokes_actuales)
        L_atras    = _evaluar_stroke(v_atras,    canvas, imagen_objetivo, n_strokes_actuales)

        # Diferencia finita centrada
        grad[i] = (L_adelante - L_atras) / (2.0 * eps)

    return grad


# ──────────────────────────────────────────
# Random Search para inicialización
# ──────────────────────────────────────────
def random_search(canvas, imagen_objetivo, n_strokes_actuales, n_candidatos=50):
    """
    Exploramos n_candidatos strokes aleatorios y devolvemos el que más
    reduce la pérdida.
    Esta es la fase de exploración global.

    Generamos múltiples soluciones aleatorias independientes en el espacio
    de búsqueda y nos quedamos con la que produce la menor pérdida.

    Esto nos da un buen punto de partida antes del refinamiento local con BFGS.
    No hay un "vecindario" fijo porque el espacio de parámetros es continuo
    (x, y, theta, r1, r2, R, G, B, alpha).

    Flujo por cada candidato:
        stroke aleatorio
        renderizar sobre copia del canvas
        calcular pérdida
        ¿es mejor que el mejor hasta ahora? -> guardar

    Al final devolvemos el stroke con la pérdida más baja encontrada.

    Args:
        canvas             : numpy array (H,W,3) — canvas actual (no se modifica)
        imagen_objetivo    : numpy array (H,W,3) — imagen original
        n_strokes_actuales : int — cuántos strokes hay ya en el canvas
        n_candidatos       : int — cuántos strokes aleatorios explorar (default 50)

    Returns:
        mejor_stroke  : Stroke — el mejor encontrado
        mejor_perdida : float  — su pérdida asociada
    """
    mejor_stroke  = None
    mejor_perdida = float('inf')  # infinito: cualquier pérdida real será menor

    for i in range(n_candidatos):

        if i % 10 == 0:
            print(f"Random Search: {i}/{n_candidatos}")
        # Generamos un stroke aleatorio con shape='ellipse'
        # Stroke.aleatorio() ya existe en modelo.py, la reutilizamos
        candidato = Stroke.aleatorio(shape=SHAPE, r_max=0.12)

        # Vemos qué pérdida tendría este candidato
        perdida_candidato = _evaluar_stroke(
            candidato.to_vector(),
            canvas,
            imagen_objetivo,
            n_strokes_actuales
        )

        # Nos quedamos con el mejor hasta el momento
        if perdida_candidato < mejor_perdida:
            mejor_perdida = perdida_candidato
            mejor_stroke  = candidato

        

    # Al terminar los n_candidatos, devolvemos el ganador
    return mejor_stroke, mejor_perdida


# ────────────────────────────────────────────────────────────
# _sufficient_decrease  (condición de Wolfe 1 — Armijo)
# ────────────────────────────────────────────────────────────

def _sufficient_decrease(f_new, f_k, grad_k, alpha, p_k, c1=1e-4):
    """
    Primera condición de Wolfe (sufficient decrease / Armijo):

        f(x + α·p) ≤ f(x) + c1·α·∇f(x)ᵀ·p

    Garantiza que el paso α produce una reducción suficiente de f.
    c1=1e-4 es el valor estándar de la literatura (Nocedal & Wright, 2006).

    Args:
        f_new  : float — f(x + α·p), valor de la función en el punto nuevo
        f_k    : float — f(x),        valor actual
        grad_k : array (9,) — gradiente en x actual
        alpha  : float — tamaño de paso candidato
        p_k    : array (9,) — dirección de descenso
        c1     : float — constante de sufficient decrease (default 1e-4)

    Returns:
        bool — True si la condición se cumple
    """
    return f_new <= f_k + c1 * alpha * np.dot(grad_k, p_k)


# ────────────────────────────────────────────────────────────
# _curvature  (condición de Wolfe 2 — curvature condition)
# ────────────────────────────────────────────────────────────

def _curvature(grad_new, grad_k, p_k, c2=0.9):
    """
    Segunda condición de Wolfe (curvature condition):

        ∇f(x + α·p)ᵀ·p ≥ c2·∇f(x)ᵀ·p

    Garantiza que el paso α no es demasiado pequeño: la pendiente
    en el punto nuevo debe ser menos negativa que en el punto actual.
    c2=0.9 es el valor estándar para métodos quasi-Newton.

    Juntas, sufficient_decrease + curvature forman las condiciones
    de Wolfe que nos garantizan convergencia global del método BFGS.

    Args:
        grad_new : array (9,) — gradiente en x + α·p
        grad_k   : array (9,) — gradiente en x actual
        p_k      : array (9,) — dirección de descenso
        c2       : float — constante de curvature (default 0.9)

    Returns:
        bool — True si la condición se cumple
    """
    return np.dot(grad_new, p_k) >= c2 * np.dot(grad_k, p_k)


# ────────────────────────────────────────────────────────────
# _busqueda_linea_wolfe  (backtracking con condiciones de Wolfe)
# ────────────────────────────────────────────────────────────

def _busqueda_linea_wolfe(x_k, f_k, grad_k, p_k, canvas, imagen_objetivo,
                          n_strokes_actuales, alpha_init=1.0, ro=0.8,
                          max_iter=20):
    """
    Búsqueda de línea con backtracking hasta satisfacer las condiciones
    de Wolfe (sufficient decrease + curvature).

    Algoritmo:
        1. Empezamos con alpha = alpha_init (= 1.0, estándar en quasi-Newton)
        2. Evaluamos f y ∇f en x + alpha·p
        3. Si sufficient_decrease Y curvature se cumplen → aceptamos alpha
        4. Si no → reducimos alpha *= ro y repetimos
        5. Si alpha se vuelve muy pequeño → aceptamos de todas formas
           (salvaguarda numérica para no quedar atrapados)

    Args:
        x_k                : array (9,) — punto actual
        f_k                : float — f(x_k)
        grad_k             : array (9,) — ∇f(x_k)
        p_k                : array (9,) — dirección de descenso (-B·∇f)
        canvas             : numpy array (H,W,3)
        imagen_objetivo    : numpy array (H,W,3)
        n_strokes_actuales : int
        alpha_init         : float — alpha inicial (default 1.0)
        ro                 : float — factor de reducción (default 0.8)
        max_iter           : int   — máximo de reducciones (default 50)

    Returns:
        alpha    : float      — tamaño de paso aceptado
        x_new    : array (9,) — x_k + alpha·p_k  (ya clipeado a BOUNDS)
        f_new    : float      — f(x_new)
        grad_new : array (9,) — ∇f(x_new)
    """
    alpha = alpha_init

    for _ in range(max_iter):
        x_new    = np.clip(x_k + alpha * p_k,
                           [b[0] for b in BOUNDS],
                           [b[1] for b in BOUNDS])
        f_new    = _evaluar_stroke(x_new, canvas, imagen_objetivo, n_strokes_actuales)
        grad_new = _gradiente_numerico(x_new, canvas, imagen_objetivo, n_strokes_actuales)

        if (_sufficient_decrease(f_new, f_k, grad_k, alpha, p_k) and
                _curvature(grad_new, grad_k, p_k)):
            return alpha, x_new, f_new, grad_new

        alpha *= ro

    # Salvaguarda: si nunca se satisfacen ambas, devolvemos el último punto
    return alpha, x_new, f_new, grad_new


# ────────────────────────────────────────────
# optimizar_bfgs  (refinamiento Quasi-Newton)
# ────────────────────────────────────────────

def optimizar_bfgs(canvas, imagen_objetivo, stroke_inicial, n_strokes_actuales,
                   tao=1e-3, max_iter=30):
    """
    Refinamos los parámetros de un stroke usando BFGS (Quasi-Newton) con
    búsqueda de línea manual por condiciones de Wolfe.

    Partimos del stroke_inicial que viene de random_search y ajustamos sus
    9 parámetros para bajar la pérdida lo más posible.

    Analogía:
        random_search → caminar a ciegas hasta encontrar un valle
        BFGS          → desde ese valle, usar el mapa topográfico
                        (Hessiano aproximado) para bajar al punto más bajo

    Algoritmo BFGS (Nocedal & Wright, 2006):
        1. Inicializamos B = I  (aproximación del Hessiano inverso)
        2. Calculamos dirección de descenso:  p = -B · ∇f
        3. Buscamos alpha con condiciones de Wolfe
        4. Actualizamos posición:  x ← x + alpha·p  (con clip a BOUNDS)
        5. Calculamos s = x_nuevo - x_viejo
                       y = ∇f_nuevo - ∇f_viejo
        6. Actualizamos B con fórmula BFGS:
               ρ   = 1 / (yᵀs)
               B ← (I - ρ·s·yᵀ) · B · (I - ρ·y·sᵀ) + ρ·s·sᵀ
        7. Repetimos hasta ||∇f|| < tao o max_iter

    ¿Por qué B aproxima el Hessiano inverso y no el Hessiano?
    Porque necesitamos resolver  p = -H⁻¹·∇f  en cada paso.
    Almacenar B = H⁻¹ directamente evita resolver un sistema lineal
    en cada iteración — solo hacemos un producto matriz-vector B·∇f.

    Los bounds se aplican con np.clip después de cada actualización,
    garantizando que los parámetros permanezcan en rangos válidos
    (posición en [0,1], radios en [0.01,0.30], etc.)

    Args:
        canvas             : numpy array (H,W,3) — canvas actual
        imagen_objetivo    : numpy array (H,W,3) — imagen original
        stroke_inicial     : Stroke — punto de partida (viene de random_search)
        n_strokes_actuales : int — strokes ya en el canvas
        tao                : float — tolerancia del gradiente (default 1e-3)
        max_iter           : int   — máximo de iteraciones (default 30)

    Returns:
        Stroke — stroke con parámetros refinados (mejor que el inicial)
    """
    # ── Inicialización ──────────────────────────────────────────────────
    x    = np.array(stroke_inicial.to_vector(), dtype=np.float64)
    n    = len(x)                  # 9 parámetros
    B    = np.eye(n)               # aproximación inicial del Hessiano inverso = I
    grad = _gradiente_numerico(x, canvas, imagen_objetivo, n_strokes_actuales)

    # ── Loop BFGS ───────────────────────────────────────────────────────
    for k in range(max_iter):

        print(f"BFGS iter {k+1}/{max_iter}")

        # Criterio de paro: gradiente suficientemente pequeño
        if np.linalg.norm(grad) < tao:
            break

        # Paso 1: dirección de descenso  p = -B · ∇f
        p = -B @ grad

        # Salvaguarda numérica: si p contiene NaN/Inf, reseteamos B = I
        if not np.all(np.isfinite(p)):
            B = np.eye(n)
            p = -grad

        # Paso 2: búsqueda de línea con condiciones de Wolfe
        f_k = _evaluar_stroke(x, canvas, imagen_objetivo, n_strokes_actuales)
        _, x_new, _, grad_new = _busqueda_linea_wolfe(
            x, f_k, grad, p, canvas, imagen_objetivo, n_strokes_actuales
        )

        # Paso 3: calculamos s e y para la actualización BFGS
        s  = x_new - x              # desplazamiento en el espacio de parámetros
        y  = grad_new - grad        # cambio de gradiente
        ys = np.dot(y, s)           # producto escalar yᵀs

        # Paso 4: actualización BFGS del Hessiano inverso aproximado
        # Solo si yᵀs > 0 (condición de curvatura positiva)
        # Si yᵀs ≤ 0 la actualización no sería definida positiva → la omitimos
        if ys > 1e-12:
            rho   = 1.0 / ys
            I_mat = np.eye(n)
            term1 = I_mat - rho * np.outer(s, y)
            term2 = I_mat - rho * np.outer(y, s)
            B     = term1 @ B @ term2 + rho * np.outer(s, s)

        # Paso 5: avanzamos
        x    = x_new
        grad = grad_new

    # ── Resultado ───────────────────────────────────────────────────────
    # Clip final por seguridad (la búsqueda de línea ya clipea internamente)
    x = np.clip(x, [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])

    return Stroke.from_vector(x, shape=SHAPE)


# ─────────────────────────────────────
# optimizar_stroke  (pipeline completo)
# ─────────────────────────────────────

def optimizar_stroke(canvas, imagen_objetivo, n_strokes_actuales):
    """
    Pipeline completo para encontrar el mejor stroke nuevo y agregarlo.

    Coordina random_search y optimizar_bfgs.

    Flujo:
        canvas actual
        random_search() <- exploración: 50 candidatos aleatorios
        mejor candidato
        optimizar_bfgs() <- refinamiento: ajuste fino con BFGS manual
        stroke final optimizado

    Args:
        canvas             : numpy array (H,W,3) — canvas actual
        imagen_objetivo    : numpy array (H,W,3) — imagen original
        n_strokes_actuales : int — cuántos strokes hay ya

    Returns:
        Stroke — el stroke final listo para agregar al canvas
    """
    # Fase 1: Exploración con Random Search
    # Generamos 50 strokes aleatorios y nos quedamos con el que menos pérdida tiene

    candidato, perdida_hill = random_search(
        canvas,
        imagen_objetivo,
        n_strokes_actuales,
        n_candidatos=50
    )

    # Fase 2: Refinamiento con BFGS
    # Partimos del candidato de Random Search y ajustamos sus 9 parámetros
    # para llegar al mínimo local más cercano

    stroke_final = optimizar_bfgs(
        canvas,
        imagen_objetivo,
        candidato,
        n_strokes_actuales
    )

    return stroke_final