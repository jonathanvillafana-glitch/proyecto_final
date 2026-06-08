import numpy as np
 
 
class Stroke:
    """
    Representa una pincelada con forma geométrica y parámetros de apariencia.
 
    Atributos (todos normalizados entre 0 y 1 salvo theta):
        x      : posición horizontal del centro  (0 = izquierda, 1 = derecha)
        y      : posición vertical del centro    (0 = arriba,    1 = abajo)
        theta  : ángulo de rotación en radianes  (0 a 2π)
        r1     : radio/semeje principal          (relativo al lado menor de la imagen)
        r2     : radio/semeje secundario         (relativo al lado menor de la imagen)
        color  : lista o array [R, G, B]         (valores entre 0 y 1)
        alpha  : opacidad                        (0 = invisible, 1 = opaco)
        shape  : tipo de figura                  ('circle', 'ellipse', 'rectangle')
    """
 
    def __init__(self, x, y, theta, r1, r2, color, alpha, shape='ellipse'):
        self.x     = float(np.clip(x,     0.0, 1.0))
        self.y     = float(np.clip(y,     0.0, 1.0))
        self.theta = float(theta % (2 * np.pi))
        self.r1    = float(np.clip(r1,    0.01, 0.30))
        self.r2    = float(np.clip(r2,    0.01, 0.30))
        self.color = np.clip(np.array(color, dtype=np.float32), 0.0, 1.0)
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.shape = shape
 
    #conversion a/desde el vector numrico 
 
    def to_vector(self):
        """
        Convierte el stroke a una lista de números para el optimizador.
 
        Orden: [x, y, theta, r1, r2, R, G, B, alpha]  → 9 valores
        El shape NO se incluye (es categórico, no numérico).
        """
        return [
            self.x,
            self.y,
            self.theta,
            self.r1,
            self.r2,
            float(self.color[0]),
            float(self.color[1]),
            float(self.color[2]),
            self.alpha,
        ]
 
    @staticmethod
    def from_vector(v, shape='ellipse'):
        """
        Reconstruye un Stroke desde una lista de 9 números.
        Inverso de to_vector().
 
        Args:
            v     : lista o array con 9 valores [x, y, theta, r1, r2, R, G, B, alpha]
            shape : tipo de figura (se pasa por separado porque es categórico)
        """
        return Stroke(
            x=v[0], y=v[1], theta=v[2],
            r1=v[3], r2=v[4],
            color=[v[5], v[6], v[7]],
            alpha=v[8],
            shape=shape,
        )
 
    @staticmethod
    def aleatorio(shape='ellipse', r_max=0.12):
        """
        Crea un Stroke con parámetros aleatorios.
        Útil para el optimizador de búsqueda aleatoria.
        """
        return Stroke(
            x=np.random.uniform(0.0, 1.0),
            y=np.random.uniform(0.0, 1.0),
            theta=np.random.uniform(0.0, 2 * np.pi),
            r1=np.random.uniform(0.01, r_max),
            r2=np.random.uniform(0.01, r_max),
            color=np.random.uniform(0.0, 1.0, size=3),
            alpha=np.random.uniform(0.1, 0.9),
            shape=shape,
        )
 
    def __repr__(self):
        r, g, b = self.color
        return (f"Stroke(shape={self.shape}, x={self.x:.3f}, y={self.y:.3f}, "
                f"theta={self.theta:.3f}, r1={self.r1:.3f}, r2={self.r2:.3f}, "
                f"color=({r:.2f},{g:.2f},{b:.2f}), alpha={self.alpha:.2f})")
 
 
#funciones de dibujo de forma 
 
def _mascara_circulo(cx, cy, r, H, W):
    """Devuelve máscara booleana (H, W) con True dentro del círculo."""
    Ys, Xs = np.ogrid[:H, :W]
    return (Xs - cx) ** 2 + (Ys - cy) ** 2 <= r ** 2
 
 
def _mascara_elipse(cx, cy, r1, r2, theta, H, W):
    """
    Devuelve máscara booleana (H, W) con True dentro de la elipse rotada.
    theta es el ángulo de rotación en radianes.
    """
    Ys, Xs = np.ogrid[:H, :W]
    dx = Xs - cx
    dy = Ys - cy
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    dx_rot =  cos_t * dx + sin_t * dy
    dy_rot = -sin_t * dx + cos_t * dy
    return (dx_rot / r1) ** 2 + (dy_rot / r2) ** 2 <= 1.0
 
 
def _mascara_rectangulo(cx, cy, r1, r2, theta, H, W):
    """
    Devuelve máscara booleana (H, W) con True dentro del rectángulo rotado.
    r1 = semiancho, r2 = semialto.
    """
    Ys, Xs = np.ogrid[:H, :W]
    dx = Xs - cx
    dy = Ys - cy
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    dx_rot = np.abs( cos_t * dx + sin_t * dy)
    dy_rot = np.abs(-sin_t * dx + cos_t * dy)
    return (dx_rot <= r1) & (dy_rot <= r2)
 
 
# funcion princioal del render 
 
def render(lista_de_strokes, H, W, canvas_base=None):
    """
    Dibuja una lista de strokes sobre un canvas.
 
    Proceso:
        1. Parte de canvas_base si se provee, o de un canvas blanco
        2. Por cada stroke aplica alpha blending:
               pixel_final = (1 - alpha) * pixel_canvas + alpha * color_stroke
        3. Devuelve el canvas como numpy array float32 en rango [0, 1]
 
    Args:
        lista_de_strokes : lista de objetos Stroke en orden de aplicación
        H                : alto de la imagen en píxeles
        W                : ancho de la imagen en píxeles
        canvas_base      : numpy array (H, W, 3) opcional. Si se pasa, dibuja
                           encima de él en lugar de empezar desde blanco.
                           Usado por optimizacion.py para evaluar candidatos
                           sin modificar el canvas principal.
 
    Returns:
        canvas : numpy array (H, W, 3) float32 (copia nueva, no modifica canvas_base)
    """
    if canvas_base is not None:
        canvas = canvas_base.copy().astype(np.float32)
    else:
        canvas = np.ones((H, W, 3), dtype=np.float32)  # fondo blanco
 
    lado_menor = min(H, W)
 
    for stroke in lista_de_strokes:
        cx = int(stroke.x  * W)
        cy = int(stroke.y  * H)
        r1 = max(int(stroke.r1 * lado_menor), 1)
        r2 = max(int(stroke.r2 * lado_menor), 1)
 
        if stroke.shape == 'circle':
            mascara = _mascara_circulo(cx, cy, r1, H, W)
        elif stroke.shape == 'ellipse':
            mascara = _mascara_elipse(cx, cy, r1, r2, stroke.theta, H, W)
        elif stroke.shape == 'rectangle':
            mascara = _mascara_rectangulo(cx, cy, r1, r2, stroke.theta, H, W)
        else:
            print(f"[render] Forma desconocida '{stroke.shape}', se omite.")
            continue
 
        # pixel_final = (1 - alpha) * pixel_canvas + alpha * color_stroke
        canvas[mascara] = (
            (1.0 - stroke.alpha) * canvas[mascara]
            + stroke.alpha * stroke.color
        )
 
    return canvas
 