import numpy as np


#MSE

def mse(original, reconstruccion):
    """
    Error Cuadrático Medio entre la imagen original y la reconstrucción.

    Fórmula: mean( (I - Î)² )

    Args:
        original       : numpy array (H, W, 3) float32 en [0, 1]
        reconstruccion : numpy array (H, W, 3) float32 en [0, 1]

    Returns:
        float entre 0 (idénticas) y 1 (máximo error)
    """
    return float(np.mean((original - reconstruccion) ** 2))



#  SSIM manual (solo numpy)

def ssim_manual(original, reconstruccion, ventana=11, C1=0.01**2, C2=0.03**2):
    """
    Índice de Similitud Estructural (SSIM) implementado solo con numpy.

    Compara luminosidad, contraste y estructura local usando ventanas
    deslizantes con un kernel gaussiano.

    Fórmula por ventana:
        SSIM = (2*mu1*mu2 + C1)(2*sigma12 + C2)
               ─────────────────────────────────
               (mu1² + mu2² + C1)(s1² + s2² + C2)

    Args:
        original       : numpy array (H, W, 3) o (H, W) float32 en [0, 1]
        reconstruccion : numpy array (H, W, 3) o (H, W) float32 en [0, 1]
        ventana        : tamaño del kernel gaussiano (default 11)
        C1, C2         : constantes de estabilidad numérica

    Returns:
        float entre 0 (completamente diferente) y 1 (idénticas)
    """
    # convertir a escala de grises si es RGB
    if original.ndim == 3:
        img1 = np.mean(original,       axis=2).astype(np.float64)
        img2 = np.mean(reconstruccion, axis=2).astype(np.float64)
    else:
        img1 = original.astype(np.float64)
        img2 = reconstruccion.astype(np.float64)

    # kernel gaussiano 20
    sigma  = 1.5
    radio  = ventana // 2
    coords = np.arange(-radio, radio + 1)
    gauss_1d = np.exp(-coords**2 / (2 * sigma**2))
    gauss_1d /= gauss_1d.sum()
    kernel = np.outer(gauss_1d, gauss_1d)   # (ventana, ventana)

    # convolucion con el kernel
    def _conv(img, k):
        """Convolución 2D válida usando suma de ventanas deslizantes."""
        H, W   = img.shape
        kH, kW = k.shape
        pH, pW = kH // 2, kW // 2
        #padding con reflexion para no perder bordes
        img_pad = np.pad(img, ((pH, pH), (pW, pW)), mode='reflect')
        out = np.zeros_like(img)
        for i in range(kH):
            for j in range(kW):
                out += k[i, j] * img_pad[i:i+H, j:j+W]
        return out

    mu1    = _conv(img1, kernel)
    mu2    = _conv(img2, kernel)

    mu1_sq  = mu1 * mu1
    mu2_sq  = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _conv(img1 * img1, kernel) - mu1_sq
    sigma2_sq = _conv(img2 * img2, kernel) - mu2_sq
    sigma12   = _conv(img1 * img2, kernel) - mu1_mu2

    # SSIM
    numerador   = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
    denominador = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    mapa_ssim = numerador / denominador
    return float(np.clip(np.mean(mapa_ssim), 0.0, 1.0))
#perdida compuesta 

def perdida_compuesta(original, reconstruccion, n_strokes, a=0.7, b=0.3, g=0.0):
    """
    Función de pérdida combinada que balancea precisión y simplicidad.

    Fórmula:
        L = a * MSE + b * (1 - SSIM) + g * n_strokes

    Args:
        original       : numpy array (H, W, 3) float32 en [0, 1]
        reconstruccion : numpy array (H, W, 3) float32 en [0, 1]
        n_strokes      : número de strokes usados hasta ahora (int)
        a              : peso del MSE              (default 0.7)
        b              : peso de (1 - SSIM)        (default 0.3)
        g              : penalización por strokes  (default 0.0)

    Returns:
        float — valor de pérdida (menor es mejor)

    Notas:
        - Empieza con g=0.0 hasta que el modelo funcione bien.
        - Aumenta g solo si quieres forzar reconstrucciones más simples.
        - a + b = 1.0 es una buena práctica para mantener la pérdida
          en rango interpretable.
    """
    error_mse  = mse(original, reconstruccion)
    error_ssim = 1.0 - ssim_manual(original, reconstruccion)
    penalizacion = g * n_strokes

    return a * error_mse + b * error_ssim + penalizacion