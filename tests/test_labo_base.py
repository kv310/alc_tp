"""
Unit tests del LABO, issue #1 (operaciones y álgebra base):
multiplicar, transponer, restri, resolver_triangular_inferior,
resolver_triangular_superior, norma.

Estrategia: los tests se escriben contra el CONTRATO de cada función
(firma + encabezado + definición matemática), no contra su implementación.
  1. Estructurales      — shapes, tipo de retorno, guards, no mutar entradas.
  2. Valores conocidos  — casos chicos calculados a mano.
  3. Propiedades        — identidades algebraicas que debe cumplir cualquier
                          implementación correcta (asociatividad, (AB)ᵀ = BᵀAᵀ...).
  4. Contra NumPy       — matrices aleatorias de varios tamaños.

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

# Tamaños (m, n, p) para los productos A(m×n)·B(n×p)
TAMANIOS_PRODUCTO = [(1, 1, 1), (2, 3, 4), (3, 3, 3), (5, 1, 4), (1, 6, 1), (7, 4, 2)]
TAMANIOS_CUADRADOS = [1, 2, 3, 5, 8]


@pytest.fixture
def rng():
    """Generador de números aleatorios con semilla fija (reproducible)."""
    return np.random.default_rng(12345)


def _triangular_bien_condicionada(rng, n, inferior):
    """Matriz triangular n×n con diagonal dominante (lejos de ser singular)."""
    M = rng.uniform(-1, 1, (n, n))
    M = np.tril(M) if inferior else np.triu(M)
    for i in range(n):
        M[i, i] = (n + 1.0) * (1 if rng.random() < 0.5 else -1)
    return M


# ======================================================================
# A — multiplicar
# ======================================================================

class TestMultiplicar:

    # --- 1. Estructurales ---

    @pytest.mark.parametrize("m,n,p", TAMANIOS_PRODUCTO)
    def test_shape_matriz_matriz(self, rng, m, n, p):
        A, B = rng.random((m, n)), rng.random((n, p))
        assert alc.multiplicar(A, B).shape == (m, p)

    @pytest.mark.parametrize("m,n", [(1, 1), (3, 2), (2, 5), (4, 4)])
    def test_shape_matriz_vector(self, rng, m, n):
        """Contrato: si B es un vector (n,), devuelve un vector (m,)."""
        A, v = rng.random((m, n)), rng.random(n)
        assert alc.multiplicar(A, v).shape == (m,)

    def test_guard_dimensiones_incompatibles(self, rng):
        with pytest.raises(AssertionError):
            alc.multiplicar(rng.random((2, 3)), rng.random((2, 3)))

    def test_guard_vector_incompatible(self, rng):
        with pytest.raises(AssertionError):
            alc.multiplicar(rng.random((2, 3)), rng.random(2))

    def test_no_modifica_entradas(self, rng):
        A, B = rng.random((3, 4)), rng.random((4, 2))
        A0, B0 = A.copy(), B.copy()
        alc.multiplicar(A, B)
        assert np.array_equal(A, A0) and np.array_equal(B, B0)

    # --- 2. Valores conocidos ---

    def test_valor_conocido_2x2(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])
        esperado = np.array([[19.0, 22.0], [43.0, 50.0]])
        assert np.allclose(alc.multiplicar(A, B), esperado, atol=ATOL)

    def test_valor_conocido_no_cuadradas(self):
        A = np.array([[1.0, 0.0, 2.0], [-1.0, 3.0, 1.0]])   # 2×3
        B = np.array([[3.0, 1.0], [2.0, 1.0], [1.0, 0.0]])   # 3×2
        esperado = np.array([[5.0, 1.0], [4.0, 2.0]])
        assert np.allclose(alc.multiplicar(A, B), esperado, atol=ATOL)

    def test_valor_conocido_matriz_vector(self):
        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        v = np.array([1.0, 0.0, -1.0])
        assert np.allclose(alc.multiplicar(A, v), [-2.0, -2.0], atol=ATOL)

    def test_entradas_enteras(self):
        """Matrices de enteros: el resultado tiene que ser el producto exacto."""
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[2, 0], [1, 2]])
        assert np.allclose(alc.multiplicar(A, B), [[4, 4], [10, 8]], atol=ATOL)

    def test_producto_interno_como_fila_por_columna(self):
        """(1×n)·(n×1) = matriz 1×1 con el producto interno."""
        a = np.array([[1.0, 2.0, 3.0]])
        b = np.array([[4.0], [5.0], [6.0]])
        assert np.allclose(alc.multiplicar(a, b), [[32.0]], atol=ATOL)

    def test_producto_externo_columna_por_fila(self):
        """(n×1)·(1×m) = matriz de rango 1 con entradas a_i·b_j."""
        a = np.array([[1.0], [2.0]])
        b = np.array([[3.0, 4.0, 5.0]])
        esperado = np.array([[3.0, 4.0, 5.0], [6.0, 8.0, 10.0]])
        assert np.allclose(alc.multiplicar(a, b), esperado, atol=ATOL)

    # --- 3. Propiedades ---

    @pytest.mark.parametrize("n", TAMANIOS_CUADRADOS)
    def test_identidad_neutro(self, rng, n):
        A = rng.random((n, n))
        I = np.eye(n)
        assert np.allclose(alc.multiplicar(A, I), A, atol=ATOL)
        assert np.allclose(alc.multiplicar(I, A), A, atol=ATOL)

    def test_matriz_nula(self, rng):
        A = rng.random((3, 4))
        assert np.allclose(alc.multiplicar(A, np.zeros((4, 2))), np.zeros((3, 2)), atol=ATOL)

    def test_asociatividad(self, rng):
        A, B, C = rng.random((3, 4)), rng.random((4, 5)), rng.random((5, 2))
        izq = alc.multiplicar(alc.multiplicar(A, B), C)
        der = alc.multiplicar(A, alc.multiplicar(B, C))
        assert np.allclose(izq, der, atol=ATOL)

    def test_distributividad(self, rng):
        A, B, C = rng.random((3, 4)), rng.random((4, 2)), rng.random((4, 2))
        assert np.allclose(alc.multiplicar(A, B + C),
                           alc.multiplicar(A, B) + alc.multiplicar(A, C), atol=ATOL)

    def test_no_conmutativo(self):
        """Sanity: para estas matrices AB ≠ BA (una implementación que
        transponga por error suele dar igual en casos simétricos)."""
        A = np.array([[1.0, 2.0], [0.0, 1.0]])
        B = np.array([[1.0, 0.0], [3.0, 1.0]])
        assert not np.allclose(alc.multiplicar(A, B), alc.multiplicar(B, A))
        assert np.allclose(alc.multiplicar(A, B), A @ B, atol=ATOL)

    def test_matriz_vector_igual_a_matriz_columna(self, rng):
        """A·v (vector) tiene que coincidir con A·V (V = v como columna n×1)."""
        A, v = rng.random((4, 3)), rng.random(3)
        assert np.allclose(alc.multiplicar(A, v),
                           alc.multiplicar(A, v.reshape(3, 1))[:, 0], atol=ATOL)

    # --- 4. Contra NumPy ---

    @pytest.mark.parametrize("m,n,p", TAMANIOS_PRODUCTO)
    def test_vs_numpy(self, rng, m, n, p):
        A, B = rng.uniform(-10, 10, (m, n)), rng.uniform(-10, 10, (n, p))
        assert np.allclose(alc.multiplicar(A, B), A @ B, atol=ATOL)

    @pytest.mark.parametrize("m,n", [(1, 1), (3, 2), (2, 5), (6, 6)])
    def test_vs_numpy_matriz_vector(self, rng, m, n):
        A, v = rng.uniform(-10, 10, (m, n)), rng.uniform(-10, 10, n)
        assert np.allclose(alc.multiplicar(A, v), A @ v, atol=ATOL)


# ======================================================================
# B — transponer
# ======================================================================

class TestTransponer:

    @pytest.mark.parametrize("m,n", [(1, 1), (1, 5), (5, 1), (3, 4), (4, 4)])
    def test_shape(self, rng, m, n):
        assert alc.transponer(rng.random((m, n))).shape == (n, m)

    def test_no_modifica_entrada(self, rng):
        A = rng.random((3, 2))
        A0 = A.copy()
        alc.transponer(A)
        assert np.array_equal(A, A0)

    def test_valor_conocido(self):
        A = np.array([[1, 2, 3], [4, 5, 6]])
        assert np.array_equal(alc.transponer(A), [[1, 4], [2, 5], [3, 6]])

    def test_conserva_valores_float(self):
        A = np.array([[0.5, -1.25], [3.75, 1e-12]])
        assert np.array_equal(alc.transponer(A), A.T)

    @pytest.mark.parametrize("m,n", [(2, 3), (4, 4), (1, 6)])
    def test_involucion(self, rng, m, n):
        """(Aᵀ)ᵀ = A."""
        A = rng.random((m, n))
        assert np.array_equal(alc.transponer(alc.transponer(A)), A)

    def test_simetrica_queda_igual(self, rng):
        M = rng.random((4, 4))
        S = M + M.T
        assert np.allclose(alc.transponer(S), S, atol=ATOL)

    def test_traspuesta_del_producto(self, rng):
        """(AB)ᵀ = BᵀAᵀ."""
        A, B = rng.random((3, 4)), rng.random((4, 2))
        izq = alc.transponer(alc.multiplicar(A, B))
        der = alc.multiplicar(alc.transponer(B), alc.transponer(A))
        assert np.allclose(izq, der, atol=ATOL)

    @pytest.mark.parametrize("m,n", [(1, 1), (2, 7), (7, 2), (5, 5)])
    def test_vs_numpy(self, rng, m, n):
        A = rng.uniform(-10, 10, (m, n))
        assert np.array_equal(alc.transponer(A), A.T)


# ======================================================================
# C — Sistemas triangulares: restri, resolver_triangular_inferior/superior
# ======================================================================

class TestTriangulares:

    # --- 1. Estructurales ---

    @pytest.mark.parametrize("n", TAMANIOS_CUADRADOS)
    def test_shape_inferior(self, rng, n):
        L = _triangular_bien_condicionada(rng, n, inferior=True)
        assert alc.resolver_triangular_inferior(L, rng.random(n)).shape == (n,)

    @pytest.mark.parametrize("n", TAMANIOS_CUADRADOS)
    def test_shape_superior(self, rng, n):
        U = _triangular_bien_condicionada(rng, n, inferior=False)
        assert alc.resolver_triangular_superior(U, rng.random(n)).shape == (n,)

    def test_no_modifica_entradas(self, rng):
        L = _triangular_bien_condicionada(rng, 4, inferior=True)
        b = rng.random(4)
        L0, b0 = L.copy(), b.copy()
        alc.resolver_triangular_inferior(L, b)
        assert np.array_equal(L, L0) and np.array_equal(b, b0)

    # --- 2. Valores conocidos ---

    def test_inferior_valor_conocido(self):
        """2x = 4 ; x + 3y = 11 ; -x + y + z = 2  →  (2, 3, 1)."""
        L = np.array([[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [-1.0, 1.0, 1.0]])
        b = np.array([4.0, 11.0, 2.0])
        assert np.allclose(alc.resolver_triangular_inferior(L, b), [2.0, 3.0, 1.0], atol=ATOL)

    def test_superior_valor_conocido(self):
        """x + 2y - z = 4 ; 4y + z = 9 ; 2z = 2  →  (1, 2, 1)."""
        U = np.array([[1.0, 2.0, -1.0], [0.0, 4.0, 1.0], [0.0, 0.0, 2.0]])
        b = np.array([4.0, 9.0, 2.0])
        assert np.allclose(alc.resolver_triangular_superior(U, b), [1.0, 2.0, 1.0], atol=ATOL)

    def test_identidad(self, rng):
        b = rng.random(5)
        assert np.allclose(alc.resolver_triangular_inferior(np.eye(5), b), b, atol=ATOL)
        assert np.allclose(alc.resolver_triangular_superior(np.eye(5), b), b, atol=ATOL)

    def test_diagonal(self):
        D = np.diag([2.0, -4.0, 0.5])
        b = np.array([2.0, 8.0, 1.0])
        esperado = np.array([1.0, -2.0, 2.0])
        assert np.allclose(alc.resolver_triangular_inferior(D, b), esperado, atol=ATOL)
        assert np.allclose(alc.resolver_triangular_superior(D, b), esperado, atol=ATOL)

    def test_un_elemento(self):
        assert np.allclose(alc.resolver_triangular_inferior(np.array([[4.0]]), np.array([2.0])), [0.5])
        assert np.allclose(alc.resolver_triangular_superior(np.array([[4.0]]), np.array([2.0])), [0.5])

    def test_b_entero(self):
        """b de enteros con solución no entera: no se puede truncar."""
        L = np.array([[2.0, 0.0], [1.0, 2.0]])
        b = np.array([1, 1])
        assert np.allclose(alc.resolver_triangular_inferior(L, b), [0.5, 0.25], atol=ATOL)

    # --- 3. Construcción sintética: x conocido → b = A x → recuperar x ---

    @pytest.mark.parametrize("n", TAMANIOS_CUADRADOS)
    def test_recupera_x_inferior(self, rng, n):
        L = _triangular_bien_condicionada(rng, n, inferior=True)
        x_verdadero = rng.uniform(-5, 5, n)
        b = L @ x_verdadero
        assert np.allclose(alc.resolver_triangular_inferior(L, b), x_verdadero, atol=ATOL)

    @pytest.mark.parametrize("n", TAMANIOS_CUADRADOS)
    def test_recupera_x_superior(self, rng, n):
        U = _triangular_bien_condicionada(rng, n, inferior=False)
        x_verdadero = rng.uniform(-5, 5, n)
        b = U @ x_verdadero
        assert np.allclose(alc.resolver_triangular_superior(U, b), x_verdadero, atol=ATOL)

    def test_restri_equivale_a_las_dos_variantes(self, rng):
        L = _triangular_bien_condicionada(rng, 4, inferior=True)
        U = _triangular_bien_condicionada(rng, 4, inferior=False)
        b = rng.random(4)
        assert np.allclose(alc.restri(L, b, True), alc.resolver_triangular_inferior(L, b), atol=ATOL)
        assert np.allclose(alc.restri(U, b, False), alc.resolver_triangular_superior(U, b), atol=ATOL)

    def test_restri_por_defecto_es_inferior(self, rng):
        L = _triangular_bien_condicionada(rng, 4, inferior=True)
        b = rng.random(4)
        assert np.allclose(alc.restri(L, b), alc.resolver_triangular_inferior(L, b), atol=ATOL)

    # --- 4. Contra NumPy ---

    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_vs_numpy(self, rng, n):
        L = _triangular_bien_condicionada(rng, n, inferior=True)
        U = _triangular_bien_condicionada(rng, n, inferior=False)
        b = rng.uniform(-10, 10, n)
        assert np.allclose(alc.resolver_triangular_inferior(L, b), np.linalg.solve(L, b), atol=ATOL)
        assert np.allclose(alc.resolver_triangular_superior(U, b), np.linalg.solve(U, b), atol=ATOL)

    # --- Casos degenerados (contrato del encabezado: devuelve None) ---

    def test_cero_en_la_diagonal_devuelve_none(self):
        L = np.array([[1.0, 0.0], [2.0, 0.0]])
        U = np.array([[0.0, 1.0], [0.0, 3.0]])
        b = np.array([1.0, 1.0])
        assert alc.resolver_triangular_inferior(L, b) is None
        assert alc.resolver_triangular_superior(U, b) is None


# ======================================================================
# D — norma
# ======================================================================

class TestNorma:

    # --- 1. Estructurales ---

    @pytest.mark.parametrize("p", [1, 2, 'inf', float('inf')])
    def test_devuelve_escalar(self, rng, p):
        """Tiene que devolver un número (no un array), para poder usarlo en
        comparaciones y divisiones sin sorpresas de broadcasting."""
        r = alc.norma(rng.random(5), p)
        assert np.ndim(r) == 0

    @pytest.mark.parametrize("p", [1, 2, 'inf'])
    def test_no_modifica_entrada(self, rng, p):
        x = rng.uniform(-1, 1, 6)
        x0 = x.copy()
        alc.norma(x, p)
        assert np.array_equal(x, x0)

    # --- 2. Valores conocidos ---

    @pytest.mark.parametrize("p,esperado", [(1, 7.0), (2, 5.0), ('inf', 4.0), (float('inf'), 4.0)])
    def test_valor_conocido_3_menos4(self, p, esperado):
        assert np.isclose(alc.norma(np.array([3.0, -4.0]), p), esperado, atol=ATOL)

    def test_valor_conocido_p3(self):
        x = np.array([1.0, 2.0, 2.0])   # (1 + 8 + 8)^(1/3)
        assert np.isclose(alc.norma(x, 3), 17.0 ** (1 / 3), atol=ATOL)

    @pytest.mark.parametrize("p", [1, 2, 'inf'])
    def test_vector_nulo(self, p):
        assert alc.norma(np.zeros(4), p) == 0

    @pytest.mark.parametrize("p", [1, 2, 'inf'])
    def test_vector_canonico(self, p):
        e = np.zeros(5)
        e[2] = 1.0
        assert np.isclose(alc.norma(e, p), 1.0, atol=ATOL)

    @pytest.mark.parametrize("p", [1, 2, 'inf'])
    def test_un_elemento_negativo(self, p):
        assert np.isclose(alc.norma(np.array([-3.5]), p), 3.5, atol=ATOL)

    def test_inf_toma_modulo_no_maximo(self):
        """‖x‖∞ = max |x_i|, no max x_i."""
        assert np.isclose(alc.norma(np.array([1.0, -9.0, 2.0]), 'inf'), 9.0, atol=ATOL)

    def test_entrada_entera(self):
        assert np.isclose(alc.norma(np.array([3, 4]), 2), 5.0, atol=ATOL)

    # --- 3. Propiedades ---

    @pytest.mark.parametrize("p", [1, 2, 3, 'inf'])
    def test_homogeneidad(self, rng, p):
        """‖αx‖ = |α| ‖x‖."""
        x = rng.uniform(-1, 1, 6)
        for alpha in [-3.0, 0.5, 2.0]:
            assert np.isclose(alc.norma(alpha * x, p), abs(alpha) * alc.norma(x, p), atol=ATOL)

    @pytest.mark.parametrize("p", [1, 2, 3, 'inf'])
    def test_desigualdad_triangular(self, rng, p):
        for _ in range(20):
            x, y = rng.uniform(-1, 1, 5), rng.uniform(-1, 1, 5)
            assert alc.norma(x + y, p) <= alc.norma(x, p) + alc.norma(y, p) + ATOL

    def test_orden_entre_normas(self, rng):
        """‖x‖∞ ≤ ‖x‖₂ ≤ ‖x‖₁ para todo x."""
        for _ in range(20):
            x = rng.uniform(-1, 1, 7)
            n1, n2, ninf = alc.norma(x, 1), alc.norma(x, 2), alc.norma(x, 'inf')
            assert ninf <= n2 + ATOL and n2 <= n1 + ATOL

    def test_norma2_al_cuadrado_es_producto_interno(self, rng):
        x = rng.uniform(-1, 1, 6)
        assert np.isclose(alc.norma(x, 2) ** 2, np.sum(x * x), atol=ATOL)

    # --- 4. Contra NumPy ---

    @pytest.mark.parametrize("n", [1, 3, 10, 100])
    @pytest.mark.parametrize("p,p_np", [(1, 1), (2, 2), (3, 3), ('inf', np.inf), (float('inf'), np.inf)])
    def test_vs_numpy(self, rng, n, p, p_np):
        x = rng.uniform(-100, 100, n)
        assert np.isclose(alc.norma(x, p), np.linalg.norm(x, p_np), rtol=1e-12, atol=ATOL)
