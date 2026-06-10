import os
import numpy as np
from PIL import Image
import imageio

from modelo import render
from metricas import perdida_compuesta
from optimizacion import optimizar_stroke

# ───────────────
# CONFIGURACIÓN
# ───────────────
# Parámetros

SEED        = 42          # semilla para reproducibilidad
IMAGE_SIZE  = (256, 256)  # (ancho, alto) en píxeles
N_STROKES   = 50         # cuántos strokes vamos a pintar en total
SAVE_EVERY  = 5          # cada cuántas iteraciones guardamos un frame
CARPETA_OUT = 'output/'   # aquí van los frames y el GIF

# Pesos de la función objetivo
ALPHA_LOSS  = 0.6
BETA_LOSS   = 0.35
GAMMA_LOSS  = 0.0005


# ──────────────
# cargar_imagen
# ──────────────

def cargar_imagen(path, size=IMAGE_SIZE):
    """
    Abre una imagen desde disco, la redimensiona y la convierte a
    numpy array con valores entre 0 y 1

    Las métricas (MSE, SSIM) y el alpha blending en render()
    asumen ese rango

    Args:
        path : str  — ruta a la imagen ('imagen.png')
        size : tuple (W, H) — tamaño destino en píxeles

    Returns:
        numpy array (H, W, 3) float32 con valores en [0.0, 1.0]
    """
    # PIL puede abrir prácticamente cualquier formato
    img = Image.open(path)

    # Pasamos a RGB por si la imagen tiene canal alpha o es escala de grises
    img = img.convert('RGB')

    # Redimensionamos — PIL usa (ancho, alto) = (W, H)
    img = img.resize(size, Image.LANCZOS)

    # PIL da valores 0-255 (uint8), dividimos entre 255 para quedar en [0, 1]
    array = np.array(img, dtype=np.float32) / 255.0

    # Resultado: shape (H, W, 3), dtype float32, valores en [0,1]
    return array


# ──────────────
# guardar_frame
# ──────────────

def guardar_frame(canvas, paso, carpeta=CARPETA_OUT):
    """
    Guarda el estado actual del canvas como imagen PNG numerada

    El nombre del archivo usa zero-padding a 4 dígitos:
        paso=1   -> frame_0001.png
        paso=10  -> frame_0010.png

    El zero-padding es importante para que los archivos se ordenen
    correctamente por nombre al generar el GIF

    Args:
        canvas  : numpy array (H,W,3) float32 en [0,1]
        paso    : int — número de iteración actual
        carpeta : str — carpeta donde guardar
    """
    # Creamos la carpeta si no existe
    os.makedirs(carpeta, exist_ok=True)

    # Regresamos de [0,1] a [0,255] para poder guardar como PNG
    # np.clip evita valores fuera de rango por redondeos
    img_uint8 = np.clip(canvas * 255.0, 0, 255).astype(np.uint8)

    # Nombre con 4 dígitos: f"frame_{paso:04d}.png"
    nombre = os.path.join(carpeta, f"frame_{paso:04d}.png")

    # PIL necesita un array uint8 para guardar correctamente
    Image.fromarray(img_uint8).save(nombre)


# ────────────
# generar_gif
# ────────────

def generar_gif(carpeta=CARPETA_OUT, nombre='evolucion.gif'):
    """
    imageio lee cada PNG, los acumula en una lista
    y los escribe como frames del GIF

    Args:
        carpeta : str — carpeta donde están los frames
        nombre  : str — nombre del archivo GIF resultante
    """
    # Listamos todos los PNGs de la carpeta que sean frames nuestros
    archivos = sorted([
        os.path.join(carpeta, f)
        for f in os.listdir(carpeta)
        if f.startswith('frame_') and f.endswith('.png')
    ])

    # Si no hay frames avisamos y salimos
    if not archivos:
        print("[generar_gif] No se encontraron frames en:", carpeta)
        return

    # Leemos cada frame
    frames = [imageio.imread(archivo) for archivo in archivos]

    # Guardamos el GIF
    # duration=0.05 -> cada frame dura 0.05 segundos (20 fps)
    # loop=0 -> el GIF se repite infinitamente
    ruta_gif = os.path.join(carpeta, nombre)
    imageio.mimsave(ruta_gif, frames, duration=0.05, loop=0)

    print(f"[generar_gif] GIF guardado en: {ruta_gif} ({len(frames)} frames)")


# ─────────────────────────
# main  — el loop principal
# ─────────────────────────

def main():
    """
    Loop principal del experimento

    Coordina todo: carga la imagen, inicializa el canvas,
    optimiza strokes uno por uno, guarda frames y genera el GIF
    """

    # Fijamos la semilla para que el experimento sea reproducible
    np.random.seed(SEED)

    # Paso 1: Cargar imagen objetivo
    print("[main] Cargando imagen...")
    imagen_original = cargar_imagen('imagen.png', size=IMAGE_SIZE)

    # IMAGE_SIZE = (W, H) en PIL, pero numpy lo guarda como (H, W, 3)
    # Extraemos H y W del array para no confundirnos
    H, W, _ = imagen_original.shape
    print(f"[main] Imagen cargada: {H}x{W} píxeles, rango [{imagen_original.min():.2f}, {imagen_original.max():.2f}]")

    # Paso 2: Inicializar canvas negro
    # np.zeros -> todos los píxeles en 0.0 = negro
    canvas = np.zeros((H, W, 3), dtype=np.float32)

    # Paso 3: Crear carpeta de salida
    os.makedirs(CARPETA_OUT, exist_ok=True)

    # Paso 4: Abrir archivo de log
    # 'w' crea el archivo nuevo (o sobreescribe si ya existe)
    ruta_log = os.path.join(CARPETA_OUT, 'logs_convergencia.txt')
    log = open(ruta_log, 'w', encoding='utf-8')
    log.write("iteracion,perdida\n")  # encabezado CSV para graficar después
    print(f"[main] Log abierto en: {ruta_log}")

    # Paso 5: Loop principal
    print(f"[main] Iniciando loop: {N_STROKES} strokes...\n")

    for t in range(1, N_STROKES + 1):

        # PASO 5a: Optimizar el siguiente stroke
        # optimizar_stroke hace internamente: Random Search -> BFGS
        stroke_nuevo = optimizar_stroke(canvas, imagen_original, t - 1)
        # t-1 porque antes de agregar este stroke hay t-1 strokes en el canvas

        # PASO 5b: Agregar el stroke al canvas
        # render() recibe el stroke nuevo y el canvas actual como base
        # devuelve un canvas nuevo sin modificar el original
        canvas = render(
            [stroke_nuevo],   # solo el stroke nuevo
            H, W,
            canvas_base=canvas
        )

        # PASO 5c: Calcular pérdida actual
        perdida_actual = perdida_compuesta(
            imagen_original,
            canvas,
            t,                # ahora hay t strokes en el canvas
            a=ALPHA_LOSS,
            b=BETA_LOSS,
            g=GAMMA_LOSS
        )

        # PASO 5d: Escribir en el log
        # CSV: "1,0.4321" para poder graficarlo fácilmente después
        log.write(f"{t},{perdida_actual:.6f}\n")
        log.flush()

        # PASO 5e: Imprimir progreso en consola cada 10 iteraciones
        if t % 10 == 0 or t == 1:
            print(f"  iteracion {t:4d}/{N_STROKES} | perdida: {perdida_actual:.6f}")

        # PASO 5f: Guardar frame cada SAVE_EVERY iteraciones
        if t % SAVE_EVERY == 0:
            guardar_frame(canvas, t)
            print(f"  [frame guardado] frame_{t:04d}.png")

    # Paso 6: Cerrar el log
    log.close()
    print(f"\n[main] Log cerrado")

    # Paso 7: Guardar imagen final
    ruta_final = os.path.join(CARPETA_OUT, 'resultado_final.png')
    img_final  = np.clip(canvas * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img_final).save(ruta_final)
    print(f"[main] Resultado final guardado en: {ruta_final}")

    # Paso 8: Generar GIF
    print("[main] Generando GIF...")
    generar_gif(carpeta=CARPETA_OUT, nombre='evolucion.gif')

    print("\n[main] Experimento completado")
    print(f"  Frames:   {CARPETA_OUT}frame_XXXX.png")
    print(f"  GIF:      {CARPETA_OUT}evolucion.gif")
    print(f"  Log:      {ruta_log}")
    print(f"  Final:    {ruta_final}")



# PUNTO DE ENTRADA

if __name__ == '__main__':
    main()