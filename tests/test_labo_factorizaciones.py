"""
Unit tests del LABO, issue #2 (factorizaciones matriciales):
inversa, cholesky, qr, autovalores, svd, producto_cruz
(y sus auxiliares: calculaLU, calculaLDV, esSDP, esSimetrica, metpot2k).

Estrategia: los tests se escriben contra el CONTRATO de cada función
(firma + encabezado + definición matemática), no contra su implementación.
Como las factorizaciones no son únicas (signos de QR, autovectores, SVD),
se verifican las PROPIEDADES que las definen (Q ortogonal, R triangular,
A = QR, A·S = S·D, ...) y no la igualdad elemento a elemento con NumPy.
  1. Estructurales           — shapes, tipo de retorno, casos inválidos.
  2. Valores conocidos       — casos chicos calculados a mano.
  3. Propiedades             — lo que define a cada factorización.
  4. Construcción sintética  — los usos del TP: K desde B (Zhang) y
                               H desde Q·h = 0 (DLT).
  5. Contra NumPy            — matrices aleatorias de varios tamaños.

Restricción de librerías del enunciado NO aplica acá: los tests pueden
usar @, np.linalg, etc.
"""

import os
import sys
import numpy as np
import pytest

# Importar alc.py desde la raíz del repo, independiente del cwd
ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
import alc

ATOL = 1e-9
ATOL_ITERATIVO = 1e-6   # para métodos iterativos (potencia / autovalores)

TAMANIOS_CUADRADOS = [1, 2, 3, 5, 8]


@pytest.fixture
def rng():
    """Generador de números aleatorios con semilla fija (reproducible)."""
    return np.random.default_rng(2026)


@pytest.fixture(autouse=True)
def semilla_global():
    """Fija también la semilla global de NumPy, por si alguna función
    usa np.random internamente (así los tests son reproducibles)."""
    np.random.seed(0)


# ----------------------------------------------------------------------
# Generadores de matrices de prueba
# ----------------------------------------------------------------------

def _bien_condicionada(rng, n):
    """Matriz n×n aleatoria con diagonal dominante (inversible, pivotes ≠ 0)."""
    return rng.uniform(-1, 1, (n, n)) + (n + 1.0) * np.eye(n)


def _sdp(rng, n):
    """Matriz simétrica definida positiva n×n."""
    M = rng.uniform(-1, 1, (n, n))
    return M @ M.T + n * np.eye(n)


def _simetrica_con_autovalores(rng, autovalores):
    """Matriz simétrica Q·diag(λ)·Qᵀ con autovalores conocidos."""
    n = len(autovalores)
    Q, _ = np.linalg.qr(rng.uniform(-1, 1, (n, n)))
    return Q @ np.diag(autovalores) @ Q.T


def _es_triangular_superior(M, atol=ATOL):
    return np.allclose(np.tril(M, -1), 0, atol=atol)


def _es_triangular_inferior(M, atol=ATOL):
    return np.allclose(np.triu(M, 1), 0, atol=atol)


def _tiene_columnas_ortonormales(Q, atol=ATOL):
    return np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=atol)


def _K_realista():
    """Intrínsecos plausibles para una cámara de 1280×720."""
    return np.array([[1000.0, 0.8, 640.0],
                     [0.0, 1010.0, 360.0],
                     [0.0, 0.0, 1.0]])


def _Q_homografia(H, puntos_plano):
    """Arma Q (2n×9) de la ec. 11 proyectando puntos (U, V) con una H conocida.
    Se arma acá (no con obtenerQ2d) para que el test no dependa del Ej 3."""
    filas = []
    for U, V in puntos_plano:
        w = H @ np.array([U, V, 1.0])
        u, v = w[0] / w[2], w[1] / w[2]
        filas.append([U, V, 1, 0, 0, 0, -u * U, -u * V, -u])
        filas.append([0, 0, 0, U, V, 1, -v * U, -v * V, -v])
    return np.array(filas, dtype=float)


def _misma_direccion(x, y, atol=1e-6):
    """x e y son paralelos (iguales a menos de escala y signo)."""
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    return np.allclose(x, y, atol=atol) or np.allclose(x, -y, atol=atol)


def _svd_implementada():
    try:
        return alc.svd(np.eye(2)) is not None
    except Exception:
        return True   # si explota, que fallen los tests y se vea el error


# ======================================================================
# A — Auxiliares: esCuadrada, sonIgualesTol, esSimetrica
# ======================================================================

class TestAuxiliares:

    def test_es_cuadrada(self):
        assert alc.esCuadrada(np.zeros((3, 3)))
        assert alc.esCuadrada(np.zeros((1, 1)))
        assert not alc.esCuadrada(np.zeros((2, 3)))
        assert not alc.esCuadrada(np.zeros((3, 2)))

    def test_son_iguales_tol(self):
        assert alc.sonIgualesTol(1.0, 1.0 + 1e-9, 1e-8)
        assert not alc.sonIgualesTol(1.0, 1.1, 1e-8)
        assert alc.sonIgualesTol(-2.0, -2.0, 0.0)

    def test_es_simetrica(self, rng):
        M = rng.random((4, 4))
        assert alc.esSimetrica(M + M.T)
        assert not alc.esSimetrica(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert not alc.esSimetrica(rng.random((2, 3)))

    def test_es_simetrica_respeta_tolerancia(self):
        A = np.array([[1.0, 2.0], [2.0 + 1e-9, 1.0]])
        assert alc.esSimetrica(A, atol=1e-7)
        assert not alc.esSimetrica(A, atol=1e-12)


# ======================================================================
# B — LU y LDV
# ======================================================================

class TestLU:

    @pytest.mark.parametrize("n", TAMANIOS_CUADRADOS)
    def test_reconstruye_A(self, rng, n):
        A = _bien_condicionada(rng, n)
        L, U, _ = alc.calculaLU(A)
        assert np.allclose(L @ U, A, atol=ATOL)

    @pytest.mark.parametrize("n", TAMANIOS_CUADRADOS)
    def test_forma_de_L_y_U(self, rng, n):
        L, U, _ = alc.calculaLU(_bien_condicionada(rng, n))
        assert L.shape == (n, n) and U.shape == (n, n)
        assert _es_triangular_inferior(L) and np.allclose(np.diag(L), 1.0)
        assert _es_triangular_superior(U)

    def test_valor_conocido(self):
        A = np.array([[2.0, 1.0], [4.0, 5.0]])
        L, U, _ = alc.calculaLU(A)
        assert np.allclose(L, [[1.0, 0.0], [2.0, 1.0]], atol=ATOL)
        assert np.allclose(U, [[2.0, 1.0], [0.0, 3.0]], atol=ATOL)

    def test_cant_op_es_entero_no_negativo(self, rng):
        _, _, cant_op = alc.calculaLU(_bien_condicionada(rng, 4))
        assert int(cant_op) == cant_op and cant_op >= 0

    def test_no_cuadrada_devuelve_none(self, rng):
        L, U, _ = alc.calculaLU(rng.random((2, 3)))
        assert L is None and U is None

    def test_no_modifica_entrada(self, rng):
        A = _bien_condicionada(rng, 4)
        A0 = A.copy()
        alc.calculaLU(A)
        assert np.array_equal(A, A0)

    def test_ldv_reconstruye_A(self, rng):
        A = _bien_condicionada(rng, 5)
        L, D, V = alc.calculaLDV(A)
        assert _es_triangular_inferior(L) and np.allclose(np.diag(L), 1.0)
        assert np.allclose(D, np.diag(np.diag(D)), atol=ATOL)
        assert _es_triangular_superior(V) and np.allclose(np.diag(V), 1.0)
        assert np.allclose(L @ D @ V, A, atol=ATOL)

    def test_ldv_simetrica_V_es_L_traspuesta(self, rng):
        """Si A es simétrica, A = L·D·Lᵀ (V = Lᵀ)."""
        A = _sdp(rng, 4)
        L, D, V = alc.calculaLDV(A)
        assert np.allclose(V, L.T, atol=ATOL)


# ======================================================================
# C — inversa
# ======================================================================

class TestInversa:

    # --- 1. Estructurales ---

    @pytest.mark.parametrize("n", TAMANIOS_CUADRADOS)
    def test_shape(self, rng, n):
        assert alc.inversa(_bien_condicionada(rng, n)).shape == (n, n)

    def test_no_modifica_entrada(self, rng):
        A = _bien_condicionada(rng, 4)
        A0 = A.copy()
        alc.inversa(A)
        assert np.array_equal(A, A0)

    # --- 2. Valores conocidos ---

    def test_valor_conocido_2x2(self):
        A = np.array([[4.0, 7.0], [2.0, 6.0]])
        esperado = np.array([[0.6, -0.7], [-0.2, 0.4]])
        assert np.allclose(alc.inversa(A), esperado, atol=ATOL)

    def test_identidad(self):
        assert np.allclose(alc.inversa(np.eye(4)), np.eye(4), atol=ATOL)

    def test_diagonal(self):
        D = np.diag([2.0, -4.0, 0.5])
        assert np.allclose(alc.inversa(D), np.diag([0.5, -0.25, 2.0]), atol=ATOL)

    def test_un_elemento(self):
        assert np.allclose(alc.inversa(np.array([[4.0]])), [[0.25]], atol=ATOL)

    def test_entrada_entera(self):
        A = np.array([[2, 1], [1, 1]])
        assert np.allclose(alc.inversa(A), [[1.0, -1.0], [-1.0, 2.0]], atol=ATOL)

    def test_triangular_superior(self):
        """La inversa de una triangular superior es triangular superior."""
        U = np.array([[2.0, 1.0, 3.0], [0.0, 1.0, 4.0], [0.0, 0.0, 5.0]])
        Uinv = alc.inversa(U)
        assert _es_triangular_superior(Uinv)
        assert np.allclose(U @ Uinv, np.eye(3), atol=ATOL)

    # --- 3. Propiedades ---

    @pytest.mark.parametrize("n", TAMANIOS_CUADRADOS)
    def test_por_ambos_lados_da_identidad(self, rng, n):
        A = _bien_condicionada(rng, n)
        Ainv = alc.inversa(A)
        assert np.allclose(A @ Ainv, np.eye(n), atol=ATOL)
        assert np.allclose(Ainv @ A, np.eye(n), atol=ATOL)

    def test_inversa_de_la_inversa(self, rng):
        A = _bien_condicionada(rng, 4)
        assert np.allclose(alc.inversa(alc.inversa(A)), A, atol=ATOL)

    def test_inversa_del_producto(self, rng):
        """(AB)⁻¹ = B⁻¹A⁻¹."""
        A, B = _bien_condicionada(rng, 3), _bien_condicionada(rng, 3)
        assert np.allclose(alc.inversa(A @ B), alc.inversa(B) @ alc.inversa(A), atol=ATOL)

    def test_inversa_de_la_traspuesta(self, rng):
        """(Aᵀ)⁻¹ = (A⁻¹)ᵀ."""
        A = _bien_condicionada(rng, 4)
        assert np.allclose(alc.inversa(A.T), alc.inversa(A).T, atol=ATOL)

    # --- 4. Casos del TP ---

    def test_K_intrinseca(self):
        """Ej 3: K⁻¹ de una matriz intrínseca realista."""
        K = _K_realista()
        assert np.allclose(alc.inversa(K) @ K, np.eye(3), atol=ATOL)

    def test_homografia(self, rng):
        """Ej 4: H⁻¹ de una homografía con escalas muy distintas por columna."""
        H = np.array([[12.5, -3.1, 640.0],
                      [0.7, 4.2, 120.0],
                      [0.001, -0.0004, 1.0]])
        assert np.allclose(alc.inversa(H) @ H, np.eye(3), atol=ATOL)

    # --- 5. Contra NumPy ---

    @pytest.mark.parametrize("n", [2, 4, 7, 10])
    def test_vs_numpy(self, rng, n):
        A = _bien_condicionada(rng, n)
        assert np.allclose(alc.inversa(A), np.linalg.inv(A), atol=ATOL)

    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_vs_numpy_matriz_general(self, rng, n):
        """Matriz aleatoria sin diagonal dominante (pivotes cualquiera)."""
        A = rng.uniform(-5, 5, (n, n))
        assert np.allclose(alc.inversa(A) @ A, np.eye(n), atol=1e-8)

    # --- Casos límite ---

    def test_singular_devuelve_none(self):
        A = np.array([[1.0, 2.0], [2.0, 4.0]])
        assert alc.inversa(A) is None


# ======================================================================
# D — esSDP y cholesky
# ======================================================================

class TestCholesky:

    # --- esSDP ---

    def test_es_sdp(self, rng):
        assert alc.esSDP(_sdp(rng, 4))
        assert alc.esSDP(np.eye(3))

    def test_no_sdp_simetrica_indefinida(self):
        assert not alc.esSDP(np.array([[1.0, 2.0], [2.0, 1.0]]))   # autovalores 3 y -1

    def test_no_sdp_no_simetrica(self):
        assert not alc.esSDP(np.array([[2.0, 1.0], [0.0, 2.0]]))

    def test_no_sdp_semidefinida(self):
        assert not alc.esSDP(np.array([[1.0, 1.0], [1.0, 1.0]]))   # autovalor 0

    # --- cholesky: estructurales ---

    @pytest.mark.parametrize("n", TAMANIOS_CUADRADOS)
    def test_shape_y_triangular_inferior(self, rng, n):
        L = alc.cholesky(_sdp(rng, n))
        assert L.shape == (n, n)
        assert _es_triangular_inferior(L)

    def test_diagonal_positiva(self, rng):
        L = alc.cholesky(_sdp(rng, 5))
        assert np.all(np.diag(L) > 0)

    def test_no_modifica_entrada(self, rng):
        A = _sdp(rng, 4)
        A0 = A.copy()
        alc.cholesky(A)
        assert np.array_equal(A, A0)

    # --- valores conocidos ---

    def test_valor_conocido_2x2(self):
        A = np.array([[4.0, 2.0], [2.0, 5.0]])
        assert np.allclose(alc.cholesky(A), [[2.0, 0.0], [1.0, 2.0]], atol=ATOL)

    def test_valor_conocido_3x3(self):
        A = np.array([[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]])
        esperado = np.array([[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]])
        assert np.allclose(alc.cholesky(A), esperado, atol=ATOL)

    def test_diagonal(self):
        assert np.allclose(alc.cholesky(np.diag([4.0, 9.0, 0.25])), np.diag([2.0, 3.0, 0.5]), atol=ATOL)

    # --- propiedades ---

    @pytest.mark.parametrize("n", TAMANIOS_CUADRADOS)
    def test_reconstruye_A(self, rng, n):
        A = _sdp(rng, n)
        L = alc.cholesky(A)
        assert np.allclose(L @ L.T, A, atol=ATOL)

    @pytest.mark.parametrize("c", [1e6, 1e-6])
    def test_escala(self, rng, c):
        """cholesky(c·A) = √c · cholesky(A): ser SDP no depende de la escala."""
        A = _sdp(rng, 3)
        L_escalada = alc.cholesky(c * A)
        assert L_escalada is not None
        assert np.allclose(L_escalada, np.sqrt(c) * alc.cholesky(A), rtol=1e-9, atol=0)

    # --- casos inválidos ---

    def test_indefinida_devuelve_none(self):
        assert alc.cholesky(np.array([[1.0, 2.0], [2.0, 1.0]])) is None

    def test_no_simetrica_devuelve_none(self):
        assert alc.cholesky(np.array([[4.0, 1.0], [0.0, 4.0]])) is None

    # --- construcción sintética: Zhang (Ej 3) ---

    def test_zhang_recupera_K_desde_B(self):
        """B = K⁻ᵀK⁻¹, normalizada (como sale del núcleo de A en Zhang, ‖b‖=1).
        Cholesky: B = L·Lᵀ  ⇒  K = (Lᵀ)⁻¹ / K33."""
        K = _K_realista()
        Kinv = np.linalg.inv(K)
        B = Kinv.T @ Kinv
        b = np.array([B[0, 0], B[0, 1], B[1, 1], B[0, 2], B[1, 2], B[2, 2]])
        B = B / np.linalg.norm(b)
        L = alc.cholesky(B)
        assert L is not None
        K_rec = np.linalg.inv(L.T)
        K_rec = K_rec / K_rec[2, 2]
        assert np.allclose(K_rec, K, rtol=1e-6)

    # --- contra NumPy ---

    @pytest.mark.parametrize("n", [2, 4, 7, 10])
    def test_vs_numpy(self, rng, n):
        """Con diagonal positiva, Cholesky es única: tiene que dar igual a NumPy."""
        A = _sdp(rng, n)
        assert np.allclose(alc.cholesky(A), np.linalg.cholesky(A), atol=ATOL)


# ======================================================================
# E — qr
# ======================================================================

class TestQR:

    # --- Householder ('RH', por defecto) ---

    @pytest.mark.parametrize("n", TAMANIOS_CUADRADOS)
    def test_rh_cuadrada(self, rng, n):
        A = rng.uniform(-5, 5, (n, n))
        Q, R = alc.qr(A)
        assert Q.shape == (n, n) and R.shape == (n, n)
        assert _tiene_columnas_ortonormales(Q)
        assert _es_triangular_superior(R)
        assert np.allclose(Q @ R, A, atol=ATOL)

    @pytest.mark.parametrize("m,n", [(4, 2), (6, 3), (9, 9), (10, 4)])
    def test_rh_rectangular(self, rng, m, n):
        A = rng.uniform(-5, 5, (m, n))
        Q, R = alc.qr(A, 'RH')
        assert Q.shape == (m, m) and R.shape == (m, n)
        assert _tiene_columnas_ortonormales(Q)
        assert _es_triangular_superior(R)
        assert np.allclose(Q @ R, A, atol=ATOL)

    def test_rh_columna_ya_alineada(self):
        """Primera columna ya múltiplo de e1 (caso borde de Householder)."""
        A = np.array([[3.0, 1.0], [0.0, 2.0]])
        Q, R = alc.qr(A, 'RH')
        assert _tiene_columnas_ortonormales(Q)
        assert np.allclose(Q @ R, A, atol=ATOL)

    def test_rh_mas_columnas_que_filas_devuelve_none(self, rng):
        assert alc.qr(rng.random((2, 4)), 'RH') is None

    # --- Gram-Schmidt ('GS') ---

    @pytest.mark.parametrize("n", TAMANIOS_CUADRADOS)
    def test_gs_cuadrada(self, rng, n):
        A = rng.uniform(-5, 5, (n, n))
        Q, R = alc.qr(A, 'GS')
        assert _tiene_columnas_ortonormales(Q, atol=1e-8)
        assert _es_triangular_superior(R)
        assert np.allclose(Q @ R, A, atol=ATOL)

    def test_gs_rango_deficiente_descarta_columnas(self, rng):
        """Columna 2 = columna 0 + columna 1: Q tiene que tener 2 columnas."""
        A = rng.uniform(-1, 1, (3, 3))
        A[:, 2] = A[:, 0] + A[:, 1]
        Q, R = alc.qr(A, 'GS')
        assert Q.shape == (3, 2) and R.shape == (2, 3)
        assert _tiene_columnas_ortonormales(Q)
        assert np.allclose(Q @ R, A, atol=ATOL)

    def test_gs_no_cuadrada_devuelve_none(self, rng):
        assert alc.qr(rng.random((4, 2)), 'GS') is None

    # --- comunes ---

    def test_metodo_invalido_devuelve_none(self, rng):
        assert alc.qr(rng.random((3, 3)), 'XX') is None

    def test_valor_conocido_modulos(self):
        """A = [[3, 1], [4, 2]]: |R00| = ‖(3,4)‖ = 5 y |det R| = |det A| = 2."""
        A = np.array([[3.0, 1.0], [4.0, 2.0]])
        for metodo in ['RH', 'GS']:
            Q, R = alc.qr(A, metodo)
            assert np.isclose(abs(R[0, 0]), 5.0, atol=ATOL)
            assert np.isclose(abs(R[0, 0] * R[1, 1]), 2.0, atol=ATOL)

    def test_modulos_de_R_iguales_a_numpy(self, rng):
        """R es única a menos del signo de cada fila."""
        A = rng.uniform(-5, 5, (5, 5))
        _, R_np = np.linalg.qr(A)
        for metodo in ['RH', 'GS']:
            _, R = alc.qr(A, metodo)
            assert np.allclose(np.abs(R), np.abs(R_np), atol=1e-8)

    def test_no_modifica_entrada(self, rng):
        A = rng.random((4, 4))
        A0 = A.copy()
        alc.qr(A, 'RH')
        alc.qr(A, 'GS')
        assert np.array_equal(A, A0)


# ======================================================================
# F — metpot2k y autovalores
# ======================================================================

class TestMetodoPotencia:

    def test_devuelve_autopar_dominante(self, rng):
        A = _simetrica_con_autovalores(rng, [10.0, 3.0, 1.0, 0.5])
        v, lamb, k = alc.metpot2k(A)
        v = np.asarray(v).reshape(-1)
        assert np.isclose(np.linalg.norm(v), 1.0, atol=ATOL_ITERATIVO)
        assert np.isclose(lamb, 10.0, atol=ATOL_ITERATIVO)
        assert np.allclose(A @ v, lamb * v, atol=ATOL_ITERATIVO)

    def test_dominante_negativo(self, rng):
        """El dominante es el de mayor MÓDULO, aunque sea negativo."""
        A = _simetrica_con_autovalores(rng, [-8.0, 3.0, 1.0])
        v, lamb, _ = alc.metpot2k(A)
        assert np.isclose(lamb, -8.0, atol=ATOL_ITERATIVO)

    def test_diagonal(self):
        A = np.diag([1.0, 7.0, 2.0])
        v, lamb, _ = alc.metpot2k(A)
        assert np.isclose(lamb, 7.0, atol=ATOL_ITERATIVO)
        assert _misma_direccion(np.asarray(v).reshape(-1), np.array([0.0, 1.0, 0.0]))

    def test_respeta_maximo_de_iteraciones(self, rng):
        A = _simetrica_con_autovalores(rng, [1.0, 0.999, 0.5])
        _, _, k = alc.metpot2k(A, K=5)
        assert k <= 5


class TestAutovalores:

    # --- 1. Estructurales ---

    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_shapes(self, rng, n):
        S, D = alc.autovalores(_sdp(rng, n))
        assert S.shape == (n, n) and D.shape == (n, n)

    def test_no_simetrica_devuelve_none(self):
        assert alc.autovalores(np.array([[1.0, 2.0], [0.0, 3.0]])) is None

    def test_no_modifica_entrada(self, rng):
        A = _sdp(rng, 4)
        A0 = A.copy()
        alc.autovalores(A)
        assert np.array_equal(A, A0)

    # --- 2. Valores conocidos ---

    def test_diagonal(self):
        S, D = alc.autovalores(np.diag([2.0, 5.0, 1.0]))
        assert np.allclose(np.diag(D), [5.0, 2.0, 1.0], atol=ATOL_ITERATIVO)

    def test_un_elemento(self):
        """1×1: el autovalor es el propio elemento y el autovector es [1] (o [-1])."""
        S, D = alc.autovalores(np.array([[-3.5]]))
        assert S.shape == (1, 1) and D.shape == (1, 1)
        assert np.isclose(D[0, 0], -3.5)
        assert np.isclose(abs(S[0, 0]), 1.0)

    def test_valor_conocido_2x2(self):
        """[[2, 1], [1, 2]] tiene autovalores 3 y 1."""
        S, D = alc.autovalores(np.array([[2.0, 1.0], [1.0, 2.0]]))
        assert np.allclose(np.diag(D), [3.0, 1.0], atol=ATOL_ITERATIVO)

    def test_autovalores_negativos(self, rng):
        lambdas = [6.0, -4.0, 2.0, -1.0]
        S, D = alc.autovalores(_simetrica_con_autovalores(rng, lambdas))
        assert np.allclose(np.diag(D), lambdas, atol=ATOL_ITERATIVO)

    # --- 3. Propiedades ---

    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_diagonalizacion(self, rng, n):
        """S ortogonal, D diagonal y A = S·D·Sᵀ."""
        A = _sdp(rng, n)
        S, D = alc.autovalores(A)
        assert _tiene_columnas_ortonormales(S, atol=ATOL_ITERATIVO)
        assert np.allclose(D, np.diag(np.diag(D)), atol=ATOL_ITERATIVO)
        assert np.allclose(S @ D @ S.T, A, atol=ATOL_ITERATIVO)

    @pytest.mark.parametrize("n", [3, 5])
    def test_columnas_son_autovectores(self, rng, n):
        A = _sdp(rng, n)
        S, D = alc.autovalores(A)
        for i in range(n):
            assert np.allclose(A @ S[:, i], D[i, i] * S[:, i], atol=ATOL_ITERATIVO)

    def test_orden_por_modulo_decreciente(self, rng):
        A = _simetrica_con_autovalores(rng, [1.0, -7.0, 3.0, 0.5, -2.0])
        _, D = alc.autovalores(A)
        modulos = np.abs(np.diag(D))
        assert np.all(np.diff(modulos) <= ATOL_ITERATIVO)

    def test_traza_y_determinante(self, rng):
        A = _sdp(rng, 5)
        _, D = alc.autovalores(A)
        assert np.isclose(np.sum(np.diag(D)), np.trace(A), atol=ATOL_ITERATIVO)
        assert np.isclose(np.prod(np.diag(D)), np.linalg.det(A), rtol=1e-6)

    # --- 4. Construcción sintética: DLT vía QᵀQ ---

    def test_dlt_ultimo_autovector_de_QtQ_es_h(self, rng):
        """Uso en el TP: el h que minimiza ‖Qh‖ con ‖h‖=1 es el autovector
        de QᵀQ con el menor autovalor (la última columna de S).
        Coordenadas normalizadas (como después de Hartley)."""
        H = np.array([[1.2, 0.1, 0.3],
                      [-0.2, 0.9, -0.1],
                      [0.05, 0.02, 1.0]])
        puntos = rng.uniform(-1.4, 1.4, (12, 2))
        Q = _Q_homografia(H, puntos)
        S, D = alc.autovalores(Q.T @ Q)
        assert _misma_direccion(S[:, -1], H.reshape(-1), atol=1e-5)
        assert abs(D[-1, -1]) < 1e-8

    # --- 5. Contra NumPy ---

    @pytest.mark.parametrize("n", [2, 4, 6, 9])
    def test_vs_numpy(self, rng, n):
        A = _sdp(rng, n)
        _, D = alc.autovalores(A)
        assert np.allclose(np.sort(np.diag(D)), np.linalg.eigvalsh(A), atol=ATOL_ITERATIVO)


# ======================================================================
# G — svd   (se saltean mientras svd no esté implementada)
# ======================================================================

@pytest.mark.skipif(not _svd_implementada(), reason="svd todavía no implementada (devuelve None)")
class TestSVD:

    # Contrato: V completa, U reducida (m × r, r = cantidad de σ no nulos)
    TAMANIOS = [(3, 3), (5, 3), (3, 5), (8, 9), (9, 8), (24, 9), (12, 12), (4, 1), (1, 4)]

    # --- 1. Estructurales ---

    @pytest.mark.parametrize("m,n", TAMANIOS)
    def test_formato(self, rng, m, n):
        """Matriz aleatoria (rango min(m, n)): U es m×min(m,n), S tiene n valores, Vt es n×n."""
        U, S, Vt = alc.svd(rng.uniform(-1, 1, (m, n)))
        assert U.shape == (m, min(m, n))
        assert S.shape == (n,)
        assert Vt.shape == (n, n)

    def test_formato_rango_deficiente(self, rng):
        """Rango 2 en una 5×3: U tiene solo 2 columnas, S y Vt siguen completas."""
        A = rng.uniform(-1, 1, (5, 2)) @ rng.uniform(-1, 1, (2, 3))
        U, S, Vt = alc.svd(A)
        assert U.shape == (5, 2)
        assert S.shape == (3,)
        assert Vt.shape == (3, 3)

    def test_no_modifica_entrada(self, rng):
        A = rng.random((4, 3))
        A0 = A.copy()
        alc.svd(A)
        assert np.array_equal(A, A0)

    # --- 2. Valores conocidos ---

    def test_diagonal(self):
        _, S, _ = alc.svd(np.diag([3.0, -5.0, 1.0]))
        assert np.allclose(S, [5.0, 3.0, 1.0], atol=ATOL_ITERATIVO)

    def test_valor_conocido(self):
        """[[3, 0], [4, 5]] tiene valores singulares √45 y √5."""
        _, S, _ = alc.svd(np.array([[3.0, 0.0], [4.0, 5.0]]))
        assert np.allclose(S, [np.sqrt(45.0), np.sqrt(5.0)], atol=ATOL_ITERATIVO)

    # --- 3. Propiedades ---

    @pytest.mark.parametrize("m,n", TAMANIOS)
    def test_valores_no_negativos_y_decrecientes(self, rng, m, n):
        _, S, _ = alc.svd(rng.uniform(-1, 1, (m, n)))
        assert np.all(S >= -ATOL_ITERATIVO)
        assert np.all(np.diff(S) <= ATOL_ITERATIVO)

    @pytest.mark.parametrize("m,n", TAMANIOS)
    def test_ortogonalidad(self, rng, m, n):
        U, _, Vt = alc.svd(rng.uniform(-1, 1, (m, n)))
        assert _tiene_columnas_ortonormales(U, atol=ATOL_ITERATIVO)
        assert _tiene_columnas_ortonormales(Vt.T, atol=ATOL_ITERATIVO)

    @pytest.mark.parametrize("m,n", TAMANIOS)
    def test_reconstruye_A(self, rng, m, n):
        """A = U_r · diag(σ_1..σ_r) · (primeras r filas de Vt)."""
        A = rng.uniform(-1, 1, (m, n))
        U, S, Vt = alc.svd(A)
        r = U.shape[1]
        assert np.allclose(U @ np.diag(S[:r]) @ Vt[:r], A, atol=ATOL_ITERATIVO)

    def test_rango_deficiente(self, rng):
        """Matriz de rango 2: el último valor singular tiene que ser ~0."""
        A = rng.uniform(-1, 1, (4, 2)) @ rng.uniform(-1, 1, (2, 3))
        _, S, _ = alc.svd(A)
        assert np.sum(S > 1e-8) == 2

    # --- 4. Construcción sintética: DLT (Ej 2, 3, 4, 5) ---

    def test_vt_ultima_fila_minimiza_norma(self, rng):
        """Vt[-1] minimiza ‖Ax‖ con ‖x‖ = 1: ningún vector unitario al azar
        puede dar un ‖Ax‖ menor."""
        A = rng.uniform(-1, 1, (12, 5))
        _, S, Vt = alc.svd(A)
        x_min = Vt[-1]
        assert np.isclose(np.linalg.norm(A @ x_min), S[-1], atol=ATOL_ITERATIVO)
        for _ in range(200):
            x = rng.uniform(-1, 1, 5)
            x /= np.linalg.norm(x)
            assert np.linalg.norm(A @ x) >= np.linalg.norm(A @ x_min) - ATOL_ITERATIVO

    def test_dlt_recupera_homografia(self, rng):
        """Q·h = 0 con h = H vectorizada por filas: Vt[-1] tiene que ser H
        (a menos de escala y signo)."""
        H = np.array([[1.2, 0.1, 0.3],
                      [-0.2, 0.9, -0.1],
                      [0.05, 0.02, 1.0]])
        Q = _Q_homografia(H, rng.uniform(-1.4, 1.4, (12, 2)))
        _, S, Vt = alc.svd(Q)
        assert _misma_direccion(Vt[-1], H.reshape(-1), atol=1e-5)

    def test_dlt_minimo_cuatro_puntos(self, rng):
        """Con 4 puntos en posición general Q es 8×9 de rango 8 y h es único."""
        H = np.array([[0.8, -0.3, 0.2],
                      [0.4, 1.1, -0.5],
                      [0.1, -0.05, 1.0]])
        puntos = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.2]])
        _, S, Vt = alc.svd(_Q_homografia(H, puntos))
        assert _misma_direccion(Vt[-1], H.reshape(-1), atol=1e-5)

    # --- 5. Contra NumPy ---

    @pytest.mark.parametrize("m,n", TAMANIOS)
    def test_valores_singulares_vs_numpy(self, rng, m, n):
        """NumPy devuelve min(m, n) valores; los n - m extra (si m < n) tienen que ser ~0."""
        A = rng.uniform(-1, 1, (m, n))
        _, S, _ = alc.svd(A)
        r = min(m, n)
        assert np.allclose(S[:r], np.linalg.svd(A, compute_uv=False), atol=ATOL_ITERATIVO)
        assert np.allclose(S[r:], 0, atol=ATOL_ITERATIVO)


# ======================================================================
# H — producto_cruz
# ======================================================================

class TestProductoCruz:

    # --- 1. Estructurales ---

    def test_shape(self, rng):
        assert alc.producto_cruz(rng.random(3), rng.random(3)).shape == (3,)

    @pytest.mark.parametrize("na,nb", [(2, 3), (3, 4), (4, 4)])
    def test_guard_solo_r3(self, rng, na, nb):
        with pytest.raises(AssertionError):
            alc.producto_cruz(rng.random(na), rng.random(nb))

    def test_no_modifica_entradas(self, rng):
        a, b = rng.random(3), rng.random(3)
        a0, b0 = a.copy(), b.copy()
        alc.producto_cruz(a, b)
        assert np.array_equal(a, a0) and np.array_equal(b, b0)

    # --- 2. Valores conocidos ---

    def test_canonicos_ciclicos(self):
        e1, e2, e3 = np.eye(3)
        assert np.allclose(alc.producto_cruz(e1, e2), e3, atol=ATOL)
        assert np.allclose(alc.producto_cruz(e2, e3), e1, atol=ATOL)
        assert np.allclose(alc.producto_cruz(e3, e1), e2, atol=ATOL)

    def test_valor_conocido(self):
        a, b = np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])
        assert np.allclose(alc.producto_cruz(a, b), [-3.0, 6.0, -3.0], atol=ATOL)

    def test_entradas_enteras(self):
        assert np.allclose(alc.producto_cruz(np.array([1, 0, 0]), np.array([0, 1, 0])), [0, 0, 1])

    # --- 3. Propiedades ---

    def test_anticonmutativo(self, rng):
        a, b = rng.random(3), rng.random(3)
        assert np.allclose(alc.producto_cruz(a, b), -alc.producto_cruz(b, a), atol=ATOL)

    def test_paralelos_da_cero(self, rng):
        a = rng.random(3)
        assert np.allclose(alc.producto_cruz(a, a), 0, atol=ATOL)
        assert np.allclose(alc.producto_cruz(a, -2.5 * a), 0, atol=ATOL)

    def test_perpendicular_a_ambos(self, rng):
        for _ in range(20):
            a, b = rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3)
            c = alc.producto_cruz(a, b)
            assert np.isclose(np.dot(c, a), 0, atol=ATOL)
            assert np.isclose(np.dot(c, b), 0, atol=ATOL)

    def test_norma_es_area(self, rng):
        """‖a×b‖² = ‖a‖²‖b‖² − (a·b)²  (identidad de Lagrange)."""
        a, b = rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3)
        c = alc.producto_cruz(a, b)
        assert np.isclose(np.dot(c, c), np.dot(a, a) * np.dot(b, b) - np.dot(a, b) ** 2, atol=ATOL)

    def test_completa_rotacion(self, rng):
        """Uso en el TP (Ej 3, Ej 6): con r1, r2 ortonormales, R = [r1 r2 r1×r2]
        es una rotación propia (RᵀR = I y det R = +1)."""
        Q, _ = np.linalg.qr(rng.uniform(-1, 1, (3, 3)))
        r1, r2 = Q[:, 0], Q[:, 1]
        R = np.column_stack([r1, r2, alc.producto_cruz(r1, r2)])
        assert np.allclose(R.T @ R, np.eye(3), atol=ATOL)
        assert np.isclose(np.linalg.det(R), 1.0, atol=ATOL)

    # --- 4. Contra NumPy ---

    def test_vs_numpy(self, rng):
        for _ in range(50):
            a, b = rng.uniform(-10, 10, 3), rng.uniform(-10, 10, 3)
            assert np.allclose(alc.producto_cruz(a, b), np.cross(a, b), atol=ATOL)
