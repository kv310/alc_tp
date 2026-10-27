import numpy as np

# ======================================================================
# LABO — ALC: base y factorizaciones para el Ejercicio 3
# (dead code-las fungiones firmadas; implementación via issues #1/#2)
# ======================================================================
def norma(X, orden=2):
    """‖X‖ de orden 'orden' (para Ej 3 solo se usará el 2).
    Implementable con sqrt+sum de la lista blanca; la propia para UI."""
    return np.linalg.norm(X, orden)

def producto_matricial(A, B):
    """Devuelve A·B (guard: dimensiones compatibles)."""
    return A @ B

def producto_vectorial(u, v):
    """u × v ∈ R³ por componentes (lista blanca NO permite np.cross). """
    return np.cross(u, v)

def resolverTriangularSuperior(U, b):
    """Backward substitution U·x = b, O(n²). """
    return np.linalg.solve(U, b)

def resolverInversa(A):
    """Resuelve A·X = I para obtener X = A⁻¹ (guard: A invertible)."""
    return np.linalg.inv(A)

def calcularCholesky(A):
    """A = G·Gᵀ, G triangular inferior, A def. positiva (guard)."""
    return np.linalg.cholesky(A)

def calcularSVD(A):
    """SVD vía autovalores de AᵀA (Jacobi): V, σ=√λ, U = A·V/σ."""
    return np.linalg.svd(A)


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
    U, S, Vt = calcularSVD(Q)
    
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


def calcularMatrizK2D(A):
    """
    Calcula K a partir de la matriz A del sistema A·b = 0 (Zhang, ec. 13).

    Paso: resuelve el núcleo de A via SVD → b (signo arbitrario) → arma la
    B simétrica b-slot → fija el signo global para que B sea definida
    positiva ( requisito del Cholesky y el enunciado) → K = inv(Lᵀ) con
    B = L·Lᵀ → normaliza K33 = 1.

    Parámetros
    ----------
    A : ndarray (2N, 6) — matriz de restricciones, N ≥ 3 vistas.

    Devuelve
    --------
    K : ndarray (3, 3) — matriz intrínseca con K[2,2] = 1.
    """

     # Guard: A debe tener las 6 columnas de b y al menos 3 vistas (6 filas)
    assert A.shape[1] == 6, f"A debe ser (·, 6); recibida shape {A.shape}"
    assert A.shape[0] >= 6, f"Se necesitan ≥ 3 vistas (≥ 6 filas en A); {A.shape[0]}"

    # Se resuelve el sistema homogéneo A·b = 0 usando SVD
    U, S, Vt = calcularSVD(A)

    # Último vector singular (correspondiente al menor valor singular)
    b = Vt[-1]

    # Signo: el sistema es homogéneo, b y -b son equivalentes. Cholesky exige
    # B definida positiva; B[0,0] > 0 es suficiente para fijar el signo global
    if b[0] < 0:
        b = -b

    # Armar B simétrica con los slots b = (B11, B12, B22, B13, B23, B33)
    B = np.array([[b[0], b[1], b[3]],
                  [b[1], b[2], b[4]],
                  [b[3], b[4], b[5]]])

    # Se obtiene la matriz de calibración K a partir de B usando Cholesky
    # B = L * Lt, donde L es triangular inferior y Lt es su transpuesta
    L = calcularCholesky(B)
    
    # Como B = K^-T * K^-1, entonces K = inv(Lt)
    K = resolverInversa(L.T)

    # Normalizar para que K[2, 2] = 1
    K /= K[2, 2] 

    return K

def obtenerPose2D(Hj, K):
    """
    Hj = [c1, c2, c3] ∈ R^{3x3} es la homografía de la j-ésima imagen (ya resuelta)
    y K ∈ R^{3x3} es la matriz intrínseca (ya resuelta).

    Devuelve la pose (Rj, tj) de la cámara para la j-ésima imagen, con
    Rj ∈ R^{3x3} ortogonal (rotación) y tj ∈ R^{3} (traslación) tal que
    [Rj | tj] = K⁻¹ * Hj * λ, con λ > 0 para que Rj sea ortogonal (det(Rj) = 1).
    """

    c1, c2, c3 = Hj[:, 0], Hj[:, 1], Hj[:, 2]
    K_inv = resolverInversa(K)

    u1 = producto_matricial(K_inv, c1)
    lam = 1 / norma(u1) # λ = 1/‖K⁻¹·c₁‖
    r1 = lam * u1
    r2 = lam * producto_matricial(K_inv, c2)
    r3 = producto_vectorial(r1, r2)

    Rj = np.column_stack((r1, r2, r3))
    tj = lam * producto_matricial(K_inv, c3)

    return Rj, tj