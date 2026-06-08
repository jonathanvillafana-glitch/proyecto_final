import numpy as np
from scipy.optimize import minimize

from modelo import Stroke, render
from metricas import perdida_compuesta


# ─────────────────────────────────────────
# BOUNDS (restricciones de cada parámetro)
# ─────────────────────────────────────────
# scipy.optimize.minimize con método L-BFGS-B necesita una lista de tuplas
# (mínimo, máximo) para cada parámetro del vector

# El vector tiene 9 posiciones
# mismo orden que Stroke.to_vector()
# [x, y, theta, r1, r2, R, G, B, alpha]

BOUNDS = [
    (0.0,    1.0),    # x — posición horizontal, 0=izq, 1=der
    (0.0,    1.0),    # y — posición vertical, 0=arr, 1=aba
    (0.0,    6.2832), # theta — rotación en radianes (0 a 2π = 6.2832)
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
# El guion bajo al inicio (_) es convención Python para indicar que esta función es interna

def _evaluar_stroke(vector, canvas, imagen_objetivo, n_strokes_actuales):
    """
    Dado un vector de 9 números, construye el stroke, lo dibuja sobre
    una copia del canvas y devuelve la pérdida resultante

    scipy.minimize la llama cientos de vecesinternamente, variando 'vector' en cada llamada 
    para encontrar el mínimo

    necesitamos EVALUAR el stroke sin modificar el canvas real
    Si modificáramos el canvas real, cada evaluación lo alteraría
    permanentemente y el optimizador se volvería loco

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
    # Nota: render() recibe (lista_strokes, H, W, canvas_base=...)
    canvas_evaluacion = render(
        [stroke],           # lista con un solo stroke nuevo
        H, W,               # dimensiones
        canvas_base=canvas  # render() hace .copy() internamente, no modifica canvas
    )

    # Paso 4: calcular la pérdida con los pesos del proyecto
    # Pasamos n_strokes_actuales + 1 porque estamos evaluando el estado con este nuevo stroke ya agregado
    perdida = perdida_compuesta(
        imagen_objetivo,
        canvas_evaluacion,
        n_strokes_actuales + 1,
        a=ALPHA_LOSS,
        b=BETA_LOSS,
        g=GAMMA_LOSS
    )

    return perdida


# ──────────────────────────────────────────
# random_search  (Hill Climbing estocástico)
# ──────────────────────────────────────────
def random_search(canvas, imagen_objetivo, n_strokes_actuales, n_candidatos=200):
    """
    Explora n_candidatos strokes aleatorios y devuelve el que más
    reduce la pérdida
    Implementa la fase de exploración (Hill Climbing)

    se llama random_search si es Hill Climbing estocástico consiste exactamente en esto:
    generar vecinos aleatorios del estado actual y quedarse con el mejor
    No hay un "vecindario" fijo como en Hill Climbing clásico porque
    el espacio de parámetros es continuo (x,y,theta,r1,r2,R,G,B,alpha)

    Flujo por cada candidato:
        stroke aleatorio
        renderizar sobre copia del canvas
        calcular pérdida
        ¿es mejor que el anterior mejor? -> guardar

    Al final devuelve el stroke con la pérdida más baja encontrada

    Args:
        canvas             : numpy array (H,W,3) — canvas actual (no se modifica)
        imagen_objetivo    : numpy array (H,W,3) — imagen original
        n_strokes_actuales : int — cuántos strokes hay ya en el canvas
        n_candidatos       : int — cuántos strokes aleatorios explorar (default 200)

    Returns:
        mejor_stroke  : Stroke — el mejor encontrado
        mejor_perdida : float  — su pérdida asociada
    """
    mejor_stroke  = None
    mejor_perdida = float('inf')  # infinito: cualquier pérdida real será menor

    for _ in range(n_candidatos):

        # Generamos un stroke aleatorio con shape='ellipse'
        # Stroke.aleatorio() ya existe en modelo.py, la reutilizamos
        candidato = Stroke.aleatorio(shape=SHAPE, r_max=0.12)

        # Evaluamos qué pérdida tendría este candidato
        perdida_candidato = _evaluar_stroke(
            candidato.to_vector(),
            canvas,
            imagen_objetivo,
            n_strokes_actuales
        )

        # Hill Climbing: solo nos quedamos si mejora (baja la pérdida)
        if perdida_candidato < mejor_perdida:
            mejor_perdida = perdida_candidato
            mejor_stroke  = candidato

    # Al terminar los n_candidatos, devolvemos el ganador
    return mejor_stroke, mejor_perdida


# ────────────────────────────────────────────
# optimizar_bfgs  (refinamiento Quasi-Newton)
# ────────────────────────────────────────────

def optimizar_bfgs(canvas, imagen_objetivo, stroke_inicial, n_strokes_actuales):
    """
    Refina los parámetros de un stroke usando L-BFGS-B (Quasi-Newton)

    Parte del stroke_inicial (que viene de random_search) y hace pequeños
    ajustes matemáticos a sus 9 parámetros para bajar la pérdida lo más
    posible. En lugar de explorar aleatorio, usa la pendiente (gradiente
    aproximado) de la función de pérdida para saber en qué dirección moverse

    Analogía:
        random_search -> caminar a ciegas hasta encontrar un valle
        L-BFGS-B ->desde ese valle, usar el mapa topográfico para llegar exactamente 
        al punto más bajo del valle

    ¿Por qué L-BFGS-B y no L-BFGS normal?
    La "B" al final significa "Bounded" = con límites
    Necesitamos que x,y E [0,1], radios > 0, etc.
    L-BFGS no puede imponer esos límites, L-BFGS-B sí

    Args:
        canvas             : numpy array (H,W,3) — canvas actual
        imagen_objetivo    : numpy array (H,W,3) — imagen original
        stroke_inicial     : Stroke — punto de partida (viene de random_search)
        n_strokes_actuales : int — strokes ya en el canvas

    Returns:
        Stroke — stroke con parámetros refinados (mejor que el inicial)
    """
    # El punto de inicio para scipy es el vector del mejor stroke de Hill Climbing
    vector_inicial = stroke_inicial.to_vector()

    # scipy.optimize.minimize es la función central de scipy para optimización
    # Le pasamos:
    #   fun -> la función que queremos minimizar (_evaluar_stroke)
    #   x0 -> el punto de inicio (vector de 9 números)
    #   args -> argumentos extra que recibe fun además de x0
    #           (canvas, imagen_objetivo, n_strokes_actuales)
    #   method -> 'L-BFGS-B' porque necesitamos bounds
    #   bounds -> lista de (min,max) para cada parámetro
    #   options -> configuración del optimizador

    resultado = minimize(
        fun    = _evaluar_stroke,
        x0     = vector_inicial,
        args   = (canvas, imagen_objetivo, n_strokes_actuales),
        method = 'L-BFGS-B',
        bounds = BOUNDS,
        options = {
            'maxiter': 100,   # máximo 100 iteraciones internas de L-BFGS-B
            'ftol'   : 1e-7,  # parar si la mejora entre iteraciones < 0.0000001
            'gtol'   : 1e-5,  # parar si el gradiente es casi cero (convergió)
        }
    )

    # resultado.x es el vector de 9 números optimizado
    # resultado.fun es la pérdida final
    # resultado.success indica si convergió bien (True/False)

    # Reconstruimos el Stroke desde el vector optimizado
    stroke_refinado = Stroke.from_vector(resultado.x, shape=SHAPE)

    return stroke_refinado


# ─────────────────────────────────────
# optimizar_stroke  (pipeline completo)
# ─────────────────────────────────────

def optimizar_stroke(canvas, imagen_objetivo, n_strokes_actuales):
    """
    Pipeline completo para encontrar el mejor stroke nuevo y agregarlo

    Internamente coordina random_search y optimizar_bfgs

    Flujo:
        canvas actual
        random_search() <- exploración: 200 candidatos aleatorios
        mejor candidato
        optimizar_bfgs() <- refinamiento: ajuste fino con L-BFGS-B
        stroke final optimizado

    Args:
        canvas             : numpy array (H,W,3) — canvas actual
        imagen_objetivo    : numpy array (H,W,3) — imagen original
        n_strokes_actuales : int — cuántos strokes hay ya

    Returns:
        Stroke — el stroke final listo para agregar al canvas
    """
    # Fase 1: Exploración con Hill Climbing
    # Genera 200 strokes aleatorios y devuelve el que menos pérdida tiene

    candidato, perdida_hill = random_search(
        canvas,
        imagen_objetivo,
        n_strokes_actuales,
        n_candidatos=200
    )

    # Fase 2: Refinamiento con L-BFGS-B 
    # Parte del candidato de Hill Climbing y ajusta sus 9 parámetros
    # matemáticamente para llegar al mínimo local más cercano

    stroke_final = optimizar_bfgs(
        canvas,
        imagen_objetivo,
        candidato,
        n_strokes_actuales
    )

    return stroke_final