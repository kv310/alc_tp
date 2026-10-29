import numpy as np

# ======================================================================
# LABO — ALC: operaciones y álgebra base (issue #1)
# ======================================================================


def multiplicar(A,B):
    """
    Producto matricial A·B (reemplaza a @ / np.dot).

    Parámetros
    ----------
    A : ndarray (m, n)
    B : ndarray (n, p), o vector (n,)

    Devuelve
    --------
    ndarray (m, p), o vector (m,) si B es un vector.
    """
    if len(B.shape) == 1:
        return multiplicar(A, B.reshape(B.shape[0], 1))[:,0]
    assert A.shape[1] == B.shape[0], "Dimensiones incompatibles para multiplicar"
    res = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            res[i,j] = sum(A[i,:] * B[:,j])
    return res


def transponer(A):
    """
    Transpuesta de A.

    Parámetros
    ----------
    A : ndarray (m, n)

    Devuelve
    --------
    ndarray (n, m) con resultado[j, i] = A[i, j].
    """
    trasMatrix = np.zeros(A.shape[::-1], dtype=A.dtype)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            trasMatrix[j][i] = A[i][j]
    return trasMatrix


def restri(A,b,inferior=True):
    """
    Resuelve A·x = b con A triangular, por sustitución hacia adelante
    (inferior) o hacia atrás (superior).

    Parámetros
    ----------
    A : ndarray (n, n) — triangular.
    b : ndarray (n,)
    inferior : bool — True si A es triangular inferior, False si es superior.

    Devuelve
    --------
    x : ndarray (n,), o None si A o b son None o A tiene un cero en la diagonal.
    """

    if A is None or b is None:
        return None
    for i in range(A.shape[0]):
        if abs(A[i][i]) < 1e-15:
            return None

    x=np.zeros(b.shape, dtype=float)
    if inferior:
        for i in range(A.shape[0]):
            _sum = 0
            for j in range(i):
                _sum+= A[i][j] * x[j]
            x[i] = (b[i]-_sum)/A[i][i]
    else:
        for i in range(A.shape[0]-1, -1, -1):
            _sum = 0
            for j in range(A.shape[1]-1, i, -1):
                _sum+= A[i][j] * x[j]
            x[i] = (b[i]-_sum)/A[i][i]

    return x


def resolver_triangular_inferior(A, b):
    """
    Resuelve A·x = b con A triangular inferior (sustitución hacia adelante).

    Parámetros
    ----------
    A : ndarray (n, n) — triangular inferior.
    b : ndarray (n,)

    Devuelve
    --------
    x : ndarray (n,), o None si A tiene un cero en la diagonal.
    """
    return restri(A, b, True)


def resolver_triangular_superior(A, b):
    """
    Resuelve A·x = b con A triangular superior (sustitución hacia atrás).

    Parámetros
    ----------
    A : ndarray (n, n) — triangular superior.
    b : ndarray (n,)

    Devuelve
    --------
    x : ndarray (n,), o None si A tiene un cero en la diagonal.
    """
    return restri(A, b, False)


def norma(X,p):
    """
    Norma vectorial p de X.

    Parámetros
    ----------
    X : ndarray (n,)
    p : número >= 1, o 'inf' / float('inf')

    Devuelve
    --------
    float — ‖X‖_p
    """
    if p == 'inf' or p == float('inf'):
        return max([abs(xi) for xi in X])
    return (sum([abs(xi)**p for xi in X]))**(1/p)


# ======================================================================
# LABO — ALC: factorizaciones matriciales (issue #2)
# ======================================================================


# ---------------------------- Auxiliares ------------------------------

def esCuadrada(aMatrix):
    """
    Indica si una matriz es cuadrada.

    Parámetros
    ----------
    aMatrix : ndarray (m, n)

    Devuelve
    --------
    bool — True si m == n.
    """
    return aMatrix.shape[0] == aMatrix.shape[1]


def sonIgualesTol(x,y,aTol):
    """
    Compara dos escalares con tolerancia absoluta.

    Parámetros
    ----------
    x, y : float
    aTol : float — tolerancia absoluta.

    Devuelve
    --------
    bool — True si |x - y| <= aTol.
    """
    return abs(float(x)-float(y)) <= aTol


def esSimetrica(aMatrix, atol=1e-7):
    """
    Indica si una matriz es simétrica (A = Aᵀ) con tolerancia.

    Parámetros
    ----------
    aMatrix : ndarray (m, n)
    atol : float — tolerancia absoluta por elemento.

    Devuelve
    --------
    bool — True si es cuadrada y |A[i, j] - A[j, i]| <= atol para todo i, j.
    """
    if not esCuadrada(aMatrix): return False
    aMatrixTras = transponer(aMatrix)
    for i in range(aMatrix.shape[0]):
        for j in range(aMatrix.shape[1]):
            if not sonIgualesTol(aMatrix[i][j],aMatrixTras[i][j],atol): return False
    return True


# ------------------------------ LU / inversa --------------------------

def calculaLU(A):
    """
    Factorización LU sin pivoteo (eliminación gaussiana): A = L·U.

    Parámetros
    ----------
    A : ndarray (n, n)

    Devuelve
    --------
    L : ndarray (n, n) — triangular inferior con unos en la diagonal.
    U : ndarray (n, n) — triangular superior.
    cant_op : int — cantidad de operaciones realizadas.
    Si A no es cuadrada o aparece un pivote nulo, devuelve (None, None, cant_op).
    """
    cant_op = 0
    if A is None:
        return None, None, cant_op
    m = A.shape[0]
    n = A.shape[1]
    Ac = np.array(A, dtype=float)

    if m != n:
        return None, None, cant_op

    for k in range(m-1):
        akk = Ac[k, k]
        if abs(akk) < 1e-15:
            return None, None, cant_op
        for i in range(k+1, m):
            lik = Ac[i, k] / akk
            Ac[i, k] = lik
            Ac[i, k+1:] = Ac[i, k+1:] - lik * Ac[k, k+1:]
            cant_op += 1 + 2 * (n - k - 1) + 2

    L = np.eye(m, dtype=Ac.dtype)
    U = np.zeros((m, n), dtype=Ac.dtype)
    for i in range(m):
        for j in range(n):
            if i > j:
                L[i][j] = Ac[i][j]
            else:
                U[i][j] = Ac[i][j]

    return L, U, cant_op


def inversa(A):
    """
    Inversa de A, resolviendo A·x_j = e_j para cada columna con LU.

    No usa pivoteo: si aparece un pivote nulo devuelve None, aunque A sea
    inversible (ej: [[0, 1], [1, 0]]).

    Parámetros
    ----------
    A : ndarray (n, n) — inversible.

    Devuelve
    --------
    ndarray (n, n) — A⁻¹, o None si la LU sin pivoteo falla (pivote nulo).
    """
    if A is None:
        return None
    I = np.eye(A.shape[0])
    inv = np.eye(A.shape[0])
    L,U,o = calculaLU(A)
    if L is None:
        return None
    for j in range(A.shape[1]):
        #colInv = resAxEqb(A,inv[:,j]) esta bueno pero recalculamos mil veces LU
        y = restri(L,I[:,j])
        x = restri(U,y,False)
        if x is None:
            return None
        inv[:,j] = x
    return inv


# ------------------------------ Cholesky ------------------------------

def calculaLDV(A):
    """
    Factorización LDV a partir de la LU: A = L·D·V.

    Parámetros
    ----------
    A : ndarray (n, n)

    Devuelve
    --------
    L : ndarray (n, n) — triangular inferior con unos en la diagonal.
    D : ndarray (n, n) — diagonal (los pivotes de U).
    V : ndarray (n, n) — triangular superior con unos en la diagonal.
    (None, None, None) si la LU falla.
    """
    if A is None:
        return None, None, None
    L,U,o = calculaLU(A)
    if(L is None):
        return None, None, None
    n = A.shape[0]
    m = A.shape[1]
    D = np.eye(n)
    V = np.eye(n)
    for i in range(n):
        D[i,i] = U[i,i]
        for j in range(i+1, m):
            V[i,j] = U[i,j]/D[i,i]
    return L,D,V


def esSDP(A,atol=1e-5):
    """
    Indica si A es simétrica definida positiva (simétrica y con
    todos los pivotes de la LDV positivos).

    Parámetros
    ----------
    A : ndarray (n, n)
    atol : float — tolerancia para la simetría y para considerar positivo un pivote.

    Devuelve
    --------
    bool
    """
    if A is None: return False
    if not esSimetrica(A,atol): return False
    L,D,V = calculaLDV(A)
    if D is None: return False
    for i in range(D.shape[0]):
        if D[i,i]<=atol:
            return False
    return True


def cholesky(A, atol=1e-10):
    """
    Factorización de Cholesky: A = L·Lᵀ, con L = L_LDV·√D.

    Parámetros
    ----------
    A : ndarray (n, n) — simétrica definida positiva.
    atol : float — tolerancia que se le pasa a esSDP.

    Devuelve
    --------
    ndarray (n, n) — L triangular inferior, o None si A no es SDP.
    """
    if A is None: return None
    if not esSDP(A, atol): return None
    L, D, V = calculaLDV(A)
    n = A.shape[0]
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            R[i][j] = L[i][j] * np.sqrt(D[j][j])
    return R


# --------------------------------- QR ---------------------------------

def QR_con_GS(A, tol = 1e-12, retorna_nops=False):
    """
    Factorización QR por Gram-Schmidt clásico. Descarta las columnas
    linealmente dependientes (norma <= tol).

    Parámetros
    ----------
    A : ndarray (n, n) — cuadrada.
    tol : float — umbral para descartar columnas dependientes.
    retorna_nops : bool — si True, también devuelve la cantidad de operaciones.

    Devuelve
    --------
    Q : ndarray (n, r) — columnas ortonormales (r = columnas independientes).
    R : ndarray (r, n) — triangular superior.
    (nops : int, solo si retorna_nops)
    None si A no es cuadrada.
    """
    if not esCuadrada(A): return None

    Q = np.zeros(A.shape, dtype = float)
    R = np.zeros(A.shape, dtype = float)
    n = A.shape[0]
    nops = 0
    r = 0

    for j in range(A.shape[0]):
        qj = A[:,j]

        for k in range(r):
            R[k,j] = sum(Q[:,k] * qj)
            nops += n + (n-1) #producto interno
            qj = qj - R[k,j]*Q[:,k]
            nops += n + n #n productos, n restas

        nqj = norma(qj, 2)
        nops += n + (n-1) + 1 #norma otra vez
        if nqj > tol:
            Q[:,r] = qj/nqj
            nops +=n #n divisiones
            R[r,j] = nqj
            r+=1

    Q = Q[:, :r]
    R = R[:r, :]

    if retorna_nops:
        return Q,R,nops
    return Q,R


def QR_con_HH(A, tol=1e-12, extras=False):
    """
    Factorización QR por reflexiones de Householder: A = Q·R.

    Parámetros
    ----------
    A : ndarray (m, n) con m >= n.
    tol : float — umbral para saltear una reflexión cuando u ≈ 0.
    extras : bool — si True, también devuelve las R y Q intermedias de cada paso.

    Devuelve
    --------
    Q : ndarray (m, m) — ortogonal.
    R : ndarray (m, n) — triangular superior.
    (dict con 'R_matrices' y 'Q_matrices', solo si extras)
    None si m < n.
    """

    m = A.shape[0]
    n = A.shape[1]
    if m<n: return None
    if extras:
        R_list = []
        Q_list = []

    R = np.array(A, dtype=float)
    Q = np.eye(m)

    for k in range(n):
        # x = columna k de R, desde la fila k hasta el final
        x = R[k:m,k]

        # alpha = -sign(x_0)·‖x‖₂ (el signo evita restar números parecidos)
        a = 1 if x[0] >= 0 else - 1
        a = -a * norma(x,2)

        # u = x - alpha·e_1 (e_1 de largo m-k: solo cambia u[0])
        e1 = np.zeros(m-k)
        e1[0] = 1
        u = x - a*e1

        # Si u ≈ 0, la columna ya está alineada con e_1 y no hace falta reflejar
        if norma(u,2) > tol:

            # u normalizado y como columna (m-k)×1, para armar el producto externo u·uᵀ
            u = u/norma(u,2)
            u = u.reshape(-1,1)

            # Reflexión H_k = I - 2·u·uᵀ, de tamaño (m-k)×(m-k)
            Hk = np.eye(m-k) - 2*multiplicar(u,transponer(u))

            # Se embebe en una identidad m×m, en el bloque inferior derecho desde (k, k)
            H_k = np.eye(m)
            H_k[k:,k:] = Hk

            # R ← H_k·R  y  Q ← Q·H_kᵀ
            R = multiplicar(H_k,R)
            Q = multiplicar(Q,transponer(H_k))

            # (si extras) guardar una copia de R y de Q de este paso
            if extras:
                R_list.append(np.array(R))
                Q_list.append(np.array(Q))

    if extras: return Q,R, {'R_matrices': R_list, 'Q_matrices': Q_list}
    return Q,R


def qr(A,metodo='RH' , tol=1e-12):
    """
    Factorización QR: A = Q·R.

    Parámetros
    ----------
    A : ndarray (m, n)
    metodo : 'RH' (Householder, admite m >= n) o 'GS' (Gram-Schmidt, solo cuadradas).
    tol : float — tolerancia del método elegido.

    Devuelve
    --------
    (Q, R) según el método, o None si el método no existe o A no es válida para él.
    """
    if metodo == 'RH':
        return QR_con_HH(A, tol)
    elif metodo == 'GS':
        return QR_con_GS(A, tol)
    return None


# ---------------------------- Autovalores -----------------------------

def metpot2k(A,tol=1e-15,K=1000):
    """
    Método de la potencia (aplicando A dos veces por iteración): aproxima
    el autovector y el autovalor de mayor módulo.

    Parámetros
    ----------
    A : ndarray (n, n)
    tol : float — corta cuando |v_nuevo · v_viejo - 1| <= tol.
    K : int — cantidad máxima de iteraciones.

    Devuelve
    --------
    v : ndarray (n, 1) — autovector dominante normalizado.
    lamb : float — autovalor dominante (vᵀ·A·v).
    k : int — iteraciones realizadas.
    """

    # Vector inicial fijo (1, 2, ..., n): no puede ser ortogonal al autovector
    # dominante salvo casos muy particulares, y hace el resultado reproducible
    v = np.arange(1, A.shape[0]+1, dtype=float).reshape(A.shape[0], 1)
    v_ = v
    for i in range(2):
        v_= multiplicar(A,v_)
        if norma(v_, 2) > 0:
            v_ = v_/norma(v_,2)
        else:
            v_ = np.zeros((v_.shape[0],1))

    e = multiplicar(transponer(v_), v)[0,0]
    k = 0
    while abs(e-1) > tol and k < K:
        v = v_

        for i in range(2):
            v_= multiplicar(A,v_)
            if norma(v_, 2) > 0:
                v_ = v_/norma(v_,2)
            else:
                v_ = np.zeros((v_.shape[0],1))

        e = multiplicar(transponer(v_),v)[0,0]
        k += 1

    lamb = multiplicar(transponer(v_), multiplicar(A,v_))[0,0]
    #err = e - 1

    return v_, lamb, k #, err


def autovalores(A, tol=1e-15, K=1000):
    """
    Diagonaliza una matriz simétrica: A = S·D·Sᵀ. En cada paso calcula el
    autovalor dominante con metpot2k, deflaciona con una reflexión de
    Householder y repite sobre la submatriz.

    Parámetros
    ----------
    A : ndarray (n, n) — simétrica.
    tol : float — tolerancia de metpot2k.
    K : int — iteraciones máximas de metpot2k.

    Devuelve
    --------
    S : ndarray (n, n) — ortogonal, con los autovectores como columnas.
    D : ndarray (n, n) — diagonal, con los autovalores de mayor a menor módulo.
    None si A no es simétrica.
    """
    if not esSimetrica(A): return None

    # Caso 1×1: ya es diagonal, su autovalor es el único elemento y su autovector es [1]
    if A.shape[0] == 1:
        return np.eye(1), np.array(A, dtype=float)

    v1, lamb1, ktrash = metpot2k(A,tol,K)

    I = np.eye(A.shape[0])
    e1=np.zeros((A.shape[0], 1))
    e1[0,0] = 1
    res1 = e1-v1
    Hv1 = I - 2*(multiplicar(res1, transponer(res1))/multiplicar(transponer(res1),res1))

    if A.shape[0] == 2:
        S = Hv1
        D = multiplicar(Hv1, multiplicar(A, transponer(Hv1)))
        D = (D + transponer(D)) / 2 #simetrizar (el redondeo la deja apenas asimetrica)

    else:
        B = multiplicar(Hv1, multiplicar(A, transponer(Hv1)))
        B = (B + transponer(B)) / 2 #simetrizar (el redondeo la deja apenas asimetrica)
        A_ = B[1:,1:]
        S_, D_ = autovalores(A_,tol,K)

        D = np.zeros((D_.shape[0]+1, D_.shape[1]+1))
        D[0,0] = lamb1
        D[1:, 1:] = D_

        S = np.eye(S_.shape[0]+1)
        S[1:, 1:] = S_
        S = multiplicar(Hv1, S)

    return S, D


# ------------------------- Pendientes del #2 --------------------------

def svd(A, tol=1e-7):
    """
    Descomposición en valores singulares: A = U·Σ·Vᵀ, calculada a partir de
    la diagonalización de AᵀA (AᵀA = V·Σ²·Vᵀ).

    Se devuelve V completa y U reducida: solo se calculan los u_i de los
    σ_i no nulos, porque u_i = A·v_i / σ_i. Para el DLT alcanza, ya que la
    solución es Vt[-1], el v del menor valor singular.

    Parámetros
    ----------
    A : ndarray (m, n)
    tol : float — un σ_i con |σ_i - 0| <= tol se considera cero (no se
          calcula su u_i). No puede ser mucho menor que 1e-8: al pasar por
          AᵀA, un σ que en teoría es 0 sale como √(error de redondeo) ≈ 1e-8·σ_1.

    Devuelve
    --------
    U : ndarray (m, r) — vectores singulares izquierdos (columnas) de los
        r valores singulares no nulos.
    S : ndarray (n,) — valores singulares de mayor a menor (incluye los nulos;
        si m < n, al menos n - m son cero).
    Vt : ndarray (n, n) — vectores singulares derechos (filas); Vt[-1] es el
         del menor valor singular (solución de mín ‖Ax‖ con ‖x‖ = 1).
    """
    n = A.shape[1]

    # Paso 1: armar AᵀA (n×n, simétrica y semidefinida positiva)
    AtA = multiplicar(transponer(A), A)

    # Paso 2: diagonalizar AᵀA = S·D·Sᵀ
    #   columnas de S = autovectores v_i, diagonal de D = autovalores λ_i = σ_i²
    #   (autovalores ya los devuelve ordenados de mayor a menor)
    S, D = autovalores(AtA)

    # Paso 3: valores singulares σ_i = √λ_i
    #   (λ_i >= 0 en teoría; un negativo chiquito es error de redondeo → 0)
    sigmas = np.zeros(n)
    for i in range(n):
        lamb = D[i, i]
        if lamb < 0:
            lamb = 0
        sigmas[i] = np.sqrt(lamb)

    # Paso 4: vectores singulares izquierdos u_i = A·v_i / σ_i,
    #   solo para los σ_i no nulos (no se puede dividir por cero)
    columnasU = []
    for i in range(n):
        if not sonIgualesTol(sigmas[i], 0, tol):
            u_i = multiplicar(A, S[:, i]) / sigmas[i]
            columnasU.append(u_i)
    U = np.array(columnasU).T

    # Paso 5: devolver en el formato de np.linalg.svd (V traspuesta)
    Vt = transponer(S)
    return U, sigmas, Vt


def producto_cruz(a, b):
    """
    Producto vectorial a × b (reemplaza a np.cross).

    Parámetros
    ----------
    a, b : ndarray (3,)

    Devuelve
    --------
    ndarray (3,) — vector perpendicular a a y a b.
    """
    assert a.shape[0] == 3 and b.shape[0] == 3, "El producto cruz está definido solo en R^3"
    res = np.zeros(3)
    res[0] = a[1]*b[2] - a[2]*b[1]
    res[1] = a[2]*b[0] - a[0]*b[2]
    res[2] = a[0]*b[1] - a[1]*b[0]
    return res



def svd_reducida(A, k="max", tol=1e-15):
    m = A.shape[0]
    n = A.shape[1]

    if m < n:
        hatV_t, hatSig, hatU_t = svd_reducida(traspuesta(A), k, tol)
        return hatU_t, hatSig, hatV_t

    AtA = multAB(traspuesta(A), A)
    S, D = diagRH(AtA, 10000)

    r = 0
    while r < n and not sonIgualesTol(D[r, r], 0, tol):
        r += 1
    if k != "max" and k < r:
        r = k

    hatSig = np.zeros(r)
    for i in range(r):
        hatSig[i] = np.sqrt(abs(D[i, i]))
    hatV = S[:, :r]

    B = multAB(A, hatV)
    hatU = np.zeros((m, r))
    for i in range(r):
        hatU[:, i] = B[:, i] / norma(B[:, i], 2)

    return hatU, hatSig, hatV