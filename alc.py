import numpy as np

# ============ Ejercicio 3 — Zhang coplanar ============
def obtenerQ2d(imgPts, objPts):
    """
    Arma la matriz Q ∈ R^{2nx9} del sistema homogéneo Q·m = 0 (ec. 11)
    para calibración coplanar: cada punto aporta 2 filas
    [U V 1 0 0 0 -uU -uV -u] y [0 0 0 U V 1 -vU -vV -v].

    Parámetros
    ----------
    imgPts : ndarray (n,2)  — proyecciones (u, v) en píxeles.
    objPts : ndarray (n,3)  — puntos del tablero (X, Y, Z) en mm (Z=0);
                                la función usa solo (X, Y) y homogeneíza
                                internamente a (U, V, 1).

    Devuelve
    --------
    Q : ndarray (2n, 9)
    """

    n_img = imgPts.shape[0]
    n_obj = objPts.shape[0]

    # Las mediciones de la imagen y las coordenadas del objeto deben coincidir
    assert n_img == n_obj, f"Cantidad de puntos de imagen ({n_img}) y objeto ({n_obj}) debe coincidir."

    n = n_img  # cantidad de puntos
    Q = np.zeros((2*n, 9))

    for i in range(n):
        u, v = imgPts[i]        # coordenadas de la imagen (u, v)
        U, V = objPts[i, :2]    # solo tomamos U, V (W=0) y homogeneizamos a (U, V, 1)

        Q[2*i]   = [U, V, 1, 0, 0, 0, -u*U, -u*V, -u]
        Q[2*i+1] = [0, 0, 0, U, V, 1, -v*U, -v*V, -v]

    return Q

def resolverDLT2D(Q):
    """
    Resuelve Q·h = 0 con ‖h‖=1 (ec. 8: mín ‖Qh‖² con norma fijada)
    y devuelve la homografía H ∈ R^{3x3} (reshape por filas).

    Parámetros
    ----------
    Q : ndarray (2n, 9) — matriz del sistema homogéneo.

    Devuelve
    --------
    H : ndarray (3, 3)
    """

    # Se resuelve el sistema homogéneo Q·h = 0 usando SVD
    U, S, Vt = np.linalg.svd(Q)
    
    # El vector h es el último vector singular (correspondiente al menor valor singular)
    h = Vt[-1]

    return h.reshape((3, 3))

def v_ij(H, i, j):
    """
    Coeficientes de la restricción c_i^T · B · c_j, lineales en
    b = (B11, B12, B22, B13, B23, B33)  (Zhang 2000, ec. v_ij·b = 0).

    Parámetros
    ----------
    H : ndarray (3, 3) — homografía de una imagen (ya resuelta).
    i, j : int — índices 0-based de COLUMNAS de H (i, j ∈ {0, 1, 2}).

    Devuelve
    --------
    v : ndarray (6,) — fila de coeficientes para la matriz A.
    """

    # 1. Extraer las columnas de H que participan en la restricción:
    #      h_i = columna i de H    (h_i[k] = H[k, i])
    #      h_j = columna j de H    (h_j[k] = H[k, j])
    h_i = H[:, i]
    h_j = H[:, j]

    # 2. Armar el vector de 6 coeficientes, en el orden de los slots de b:
    #      slot 0 (B11): h_i[0]·h_j[0]                          ← producto simple
    #      slot 1 (B12): h_i[0]·h_j[1] + h_i[1]·h_j[0]          ← cruzada simetrizada
    #      slot 2 (B22): h_i[1]·h_j[1]                          ← producto simple
    #      slot 3 (B13): h_i[0]·h_j[2] + h_i[2]·h_j[0]          ← cruzada simetrizada
    #      slot 4 (B23): h_i[1]·h_j[2] + h_i[2]·h_j[1]          ← cruzada simetrizada
    #      slot 5 (B33): h_i[2]·h_j[2]                          ← producto simple
    #    Los slots de la DIAGONAL de B (0, 2, 5)
    #    son productos simples h_i[m]·h_j[m]; los Cruzados (1, 3, 4) son la
    #    suma simétrica h_i[m]·h_j[n] + h_i[n]·h_j[m] con (m, n) el par correspondiente.
    v = np.zeros(6)
    v[0] = h_i[0] * h_j[0]
    v[1] = h_i[0] * h_j[1] + h_i[1] * h_j[0]
    v[2] = h_i[1] * h_j[1]
    v[3] = h_i[0] * h_j[2] + h_i[2] * h_j[0]
    v[4] = h_i[1] * h_j[2] + h_i[2] * h_j[1]
    v[5] = h_i[2] * h_j[2]

    return v

def armarA(Hlst):
    """
    Arma la matriz A del sistema homogéneo A·b = 0 (Zhang) para
    calibrar la cámara a partir de N homografías ya resueltas.

    Cada homografía aporta 2 filas:
      - A[2k]   = v12(H_k):  impone  c₁ᵀ B c₂ = 0        (r1 ⊥ r2)
      - A[2k+1] = v11(H_k) - v22(H_k):  impone  c₁ᵀ B c₁ = c₂ᵀ B c₂  (‖r1‖=‖r2‖)

    con b = (B11, B12, B22, B13, B23, B33)ᵀ y B = K⁻ᵀ K⁻¹ (simétrica).

    Parámetros
    ----------
    Hlst : lista de ndarray (3, 3)
        Una homografía por imagen, en el mismo orden que las capturas.

    Devuelve
    --------
    A : ndarray (2N, 6)
    """

    # 1. Cantidad de homografías N y prealocación de A con np.zeros((2N, 6)).
    N = len(Hlst)  # cantidad de homografías N
    A = np.zeros((2 * N, 6))

    # 2. Loop por cada homografía H_k (enumerate sobre Hlst, índice k):
    #    2a. A[2k]   = v_ij(H_k, 0, 1)                   → fila v12
    #    2b. A[2k+1] = v_ij(H_k, 0, 0) − v_ij(H_k, 1, 1) → fila v11 − v22
    for k in range(N):
        H_k = Hlst[k]
        A[2*k] = v_ij(H_k, 0, 1)
        A[2*k + 1] = v_ij(H_k, 0, 0) - v_ij(H_k, 1, 1)

    return A