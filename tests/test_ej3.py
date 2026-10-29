"""
Unit tests del Ejercicio 3 (Zhang coplanar): obtenerQ2d,
resolverDLT2D, v_ij, armarA.

Estrategia: datos sintéticos con ground truth (K, R, t conocidos → H
exactas → proyección de puntos) + un smoke test de reproyección con
los datos reales de calib/coplanar (si existen).

Restricción de librerías del enunciado NO aplica acá: los tests pueden
usar @, np.dot, SVD, etc.
"""

import os
import sys
import numpy as np
import pytest

# Importar alc.py desde la raíz del repo, independiente del cwd del kernel
ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
import alc

ATOL = 1e-9   # para construcciones exactas
RTOL = 1e-6   # para roundtrips con SVD (ver comentario en TestResolverDLT2D)

# ======================================================================
# Datos sintéticos (ground truth)
# ======================================================================

def _expande_b_desde_B(B):
    """Vector b = (B11, B12, B22, B13, B23, B33) desde una matriz simétrica B."""
    return np.array([B[0, 0], B[0, 1], B[1, 1], B[0, 2], B[1, 2], B[2, 2]])


def _crear_K():
    """K verdadera, plausible para 1024×576 (centro ≈ (512, 288))."""
    return np.array([[800.0, 0.5, 512.0],
                     [0.0, 810.0, 288.0],
                     [0.0, 0.0, 1.0]])


def _crear_H_j(K, r1, r2, t):
    """H = K [r1 r2 t]  (Zhang: H = λ K [r1 r2 t]; elegimos λ tal que H[2,2]=1)."""
    H = K @ np.column_stack([r1, r2, t])
    return H / H[2, 2]


def _crear_obj_coords(n_pts):
    """Coordenadas (X, Y) no colineales de esquinas internas del tablero,
    misma geometría analítica de la práctica: (30i, 30j) mm, fila-mayor."""
    X = 30.0 * (np.arange(n_pts) % 11)
    Y = 30.0 * (np.arange(n_pts) // 11)
    return X, Y


@pytest.fixture
def datos_sinteticos():
    """
    Devuelve un dict con:
      K      — intrínsecos verdaderos
      B      — B verdadera = K^{-T} K^{-1} (simétrica, def. positiva)
      b      — b verdadero expansionado
      objPts — (77, 3) esquinas del tablero (Z=0) en mm
      Hlst   — 5 homografías verdaderas, una por vista
      Q      — Q total de las 5 vistas (obtenerQ2d)
    """
    K = _crear_K()
    B = np.linalg.inv(K.T) @ np.linalg.inv(K)   # B = K^{-T} K^{-1}
    b_true = _expande_b_desde_B(B)

    n_pts = 77
    X, Y = _crear_obj_coords(n_pts)
    objPts = np.column_stack([X, Y, np.zeros(n_pts)])

    Hlst, Qfilas = [], []
    for k in range(5):
        a = 20.0 + 7.0 * k          # rotaciones distintas por vista (no degeneradas)
        c, s = np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))
        r1 = np.array([c, 0.0, -s])
        r2 = np.array([0.0, 1.0, 0.0])
        # t_z ~ 2000: mantiene las proyecciones dentro del cuadro 1024×576
        # (u ∈ 512±120, v ≈ 288±15) conservando perspectiva (w2 varía ~11%)
        t = np.array([100.0 + 10.0 * k, 50.0 - 5.0 * k, 2000.0 + 100.0 * k])
        H_k = _crear_H_j(K, r1, r2, t)                          # ← H primero,
        w = H_k @ np.vstack([X, Y, np.ones_like(X)])            # ← y proyectar CON ella
        imgPts = np.column_stack([w[0] / w[2], w[1] / w[2]])    # (u, v) por perspectiva
        Qfilas.append(alc.obtenerQ2d(imgPts, objPts))
        Hlst.append(H_k)

    Q = np.vstack(Qfilas)
    return dict(K=K, B=B, b=b_true, objPts=objPts, Hlst=Hlst, Q=Q)



# ======================================================================
# A — obtenerQ2d
# ======================================================================

class TestObtenerQ2d:

    def test_shape(self, datos_sinteticos):
        assert datos_sinteticos["Q"].shape == (2 * 77 * 5, 9)

    def test_shape_minimo(self):
        imgPts = np.array([[100.0, 200.0]])
        objPts = np.array([[30.0, 0.0, 0.0]])
        assert alc.obtenerQ2d(imgPts, objPts).shape == (2, 9)

    def test_filas_exactas_ec11(self):
        """Un punto conocido → filas exactas contra la ecuación (11)."""
        imgPts = np.array([[123.0, 45.0]])
        objPts = np.array([[60.0, 30.0, 0.0]])
        Q = alc.obtenerQ2d(imgPts, objPts)
        assert np.allclose(Q[0], [60, 30, 1, 0, 0, 0, -123*60, -123*30, -123], atol=ATOL)
        assert np.allclose(Q[1], [0, 0, 0, 60, 30, 1, -45*60, -45*30, -45], atol=ATOL)

    def test_intercalado_de_filas(self):
        """La fila (2i, 2i+1) corresponde al punto i."""
        imgPts = np.array([[1.0, 2.0], [3.0, 4.0]])
        objPts = np.array([[10.0, 20.0, 0.0], [30.0, 40.0, 0.0]])
        Q = alc.obtenerQ2d(imgPts, objPts)
        # punto 1 → filas 2 y 3: Q[2] = [U V 1 | 0 0 0 | ...]; Q[3] = [0 0 0 | U V 1 | ...]
        assert Q[2, 0] == 30.0 and Q[2, 1] == 40.0
        assert Q[3, 3] == 30.0 and Q[3, 4] == 40.0

    def test_Qh0_con_H_verdadera(self, datos_sinteticos):
        """INVARIANTE CLAVE: con los imgPts proyectados CON Hlst[0],
        Q·h_true ≈ 0 (falla si la construcción está mal)."""
        d = datos_sinteticos
        H0 = d["Hlst"][0]
        X, Y = d["objPts"][:, 0], d["objPts"][:, 1]
        w = H0 @ np.vstack([X, Y, np.ones_like(X)])
        imgPts = np.column_stack([w[0] / w[2], w[1] / w[2]])
        Q = alc.obtenerQ2d(imgPts, d["objPts"])
        assert np.allclose(Q @ H0.reshape(-1), 0.0, atol=1e-8)

    def test_rango_ocho_por_vista(self, datos_sinteticos):
        """Criterio del issue: rango 8 en el Q de UNA vista de 77 puntos.
        (El stack de N vistas tiene rango 9: cada vista aporta su propio
        vector nulo h_k, y núcleos distintos no intersectan.)"""
        d = datos_sinteticos
        H0 = d["Hlst"][0]
        X, Y = d["objPts"][:, 0], d["objPts"][:, 1]
        w = H0 @ np.vstack([X, Y, np.ones_like(X)])
        imgPts = np.column_stack([w[0] / w[2], w[1] / w[2]])
        Q_k = alc.obtenerQ2d(imgPts, d["objPts"])          # (154, 9), una vista
        sv = np.linalg.svd(Q_k, compute_uv=False)
        assert sv[7] / sv[8] > 1e3 

    def test_minimo_4_puntos_rango_8(self):
        """El issue pide rango 8 con n ≥ 4. Con 4 puntos de una homografía REAL:
        Q es (8, 9) y h_true pertenece al núcleo (unicidad garantizada por el
        teorema de la práctica: 4 puntos en posición general → rango 8).

        NOTA NUMÉRICA: el criterio de rango NO se testea via σ7/σ8 acá — para
        esta matriz 8×9 mal condicionada LAPACK dgesdd devuelve σ de la zona
        degenerada inconsistentes (σ_min≈0.5 con ‖Q·Vt[-1]‖≈0). Las DIRECCIONES
        sí son confiables (cos(Vt[-1], h)=1.0) y con ellas resolverDLT2D
        recupera H — ese es el criterio práctico. El test de σ-gap queda
        cubierto por test_rango_ocho_por_vista (154×9, numéricamente limpio)."""
        K = _crear_K()
        c, s = np.cos(np.deg2rad(30.0)), np.sin(np.deg2rad(30.0))
        r1 = np.array([c, 0.0, -s])
        r2 = np.array([0.0, 1.0, 0.0])
        t = np.array([100.0, 50.0, 2000.0])
        H = _crear_H_j(K, r1, r2, t)
        objPts = np.array([[0.0, 0.0, 0.0], [60.0, 0.0, 0.0],
                           [0.0, 30.0, 0.0], [60.0, 30.0, 0.0]])   # grilla 2×2, no colineales
        w = H @ np.vstack([objPts[:, 0], objPts[:, 1], np.ones(4)])
        imgPts = np.column_stack([w[0] / w[2], w[1] / w[2]])        # correspondencias REALES

        Q = alc.obtenerQ2d(imgPts, objPts)
        assert Q.shape == (8, 9)

        # (a) h_true ES vector nulo (test directo del núcleo, robusto)
        assert np.allclose(Q @ H.reshape(-1), 0.0, atol=1e-8)

        # (b) unicidad → rango 8 → el DLT recupera H con el mínimo de 4 puntos.
        #     (recuperación verificada empiricamente con error ~1e-9)
        H_rec = alc.resolverDLT2D(Q)
        H_rec = H_rec / H_rec[2, 2]
        assert np.allclose(H_rec, H, atol=RTOL)

    def test_ignora_Z(self):
        """Usa solo (X, Y): cambiar Z no cambia Q."""
        imgPts = np.array([[100.0, 200.0], [150.0, 250.0]])
        objPts = np.array([[30.0, 0.0, 0.0], [60.0, 30.0, 0.0]])
        Q1 = alc.obtenerQ2d(imgPts, objPts)
        otro = objPts.copy(); otro[:, 2] = 999.0
        assert np.allclose(Q1, alc.obtenerQ2d(imgPts, otro))

    def test_guard_cantidad_puntos(self):
        imgPts = np.array([[100.0, 200.0]])
        objPts = np.array([[30.0, 0.0, 0.0], [60.0, 30.0, 0.0]])
        with pytest.raises(AssertionError):
            alc.obtenerQ2d(imgPts, objPts)

    def test_con_datos_reales(self):
        """Smoke: (154, 9) con la primera imagen real de calib/coplanar."""
        archivo = os.path.join(REPO, "calib", "coplanar", "detected_corners.txt")
        if not os.path.exists(archivo):
            pytest.skip("no hay datos de calib/coplanar en esta máquina")
        fila0 = np.loadtxt(archivo, max_rows=1)
        imgPts = fila0.reshape(77, 2)
        n = 77
        X, Y = _crear_obj_coords(n)
        objPts = np.column_stack([X, Y, np.zeros(n)])
        assert alc.obtenerQ2d(imgPts, objPts).shape == (154, 9)


# ======================================================================
# B — resolverDLT2D
# ======================================================================

class TestResolverDLT2D:

    def test_shape_y_norma(self, datos_sinteticos):
        """Devuelve (3,3) y el vector h subyacente tiene norma 1."""
        H = alc.resolverDLT2D(datos_sinteticos["Q"])
        assert H.shape == (3, 3)
        assert np.isclose(np.linalg.norm(H.reshape(-1)), 1.0, atol=1e-12)

    def test_round_trip(self, datos_sinteticos):
        """Datos exactos → recupera H salvo escala en TODAS las vistas.
        Tolerancia 1e-6 (no 1e-9): la mezcla de escalas en Q (px·mm vs O(1))
        limita la precisión del vector nulo via SVD; Hartley entra en #8."""
        d = datos_sinteticos
        X, Y = d["objPts"][:, 0], d["objPts"][:, 1]
        for k, H_true in enumerate(d["Hlst"]):
            w = H_true @ np.vstack([X, Y, np.ones_like(X)])
            imgPts = np.column_stack([w[0] / w[2], w[1] / w[2]])
            H_rec = alc.resolverDLT2D(alc.obtenerQ2d(imgPts, d["objPts"]))
            H_rec = H_rec / H_rec[2, 2]
            assert np.allclose(H_rec, H_true, atol=RTOL), f"vista {k}"

    def test_reproyeccion_con_ruido(self, datos_sinteticos):
        """Ruido de 1 px → el H del DLT reproyecta con rms < 5 px."""
        rng = np.random.default_rng(46)
        d = datos_sinteticos
        H_true = d["Hlst"][0]
        X, Y = d["objPts"][:, 0], d["objPts"][:, 1]
        w = H_true @ np.vstack([X, Y, np.ones_like(X)])
        imgPts = np.column_stack([w[0] / w[2], w[1] / w[2]])
        img_ruido = imgPts + rng.normal(0.0, 1.0, imgPts.shape)
        H_rec = alc.resolverDLT2D(alc.obtenerQ2d(img_ruido, d["objPts"]))
        w = H_rec @ np.vstack([X, Y, np.ones_like(X)])
        proj = np.column_stack([w[0] / w[2], w[1] / w[2]])
        rms = np.sqrt(np.mean(np.sum((proj - imgPts) ** 2, axis=1)))
        assert rms < 5.0

    def test_con_datos_reales_reproyeccion(self):
        """Baseline del QA: imagen 1 real de calib/coplanar, max err < 1 px."""
        archivo = os.path.join(REPO, "calib", "coplanar", "detected_corners.txt")
        if not os.path.exists(archivo):
            pytest.skip("no hay datos reales en esta máquina")
        imgPts = np.loadtxt(archivo, max_rows=1).reshape(77, 2)
        X, Y = _crear_obj_coords(77)
        objPts = np.column_stack([X, Y, np.zeros(77)])
        H = alc.resolverDLT2D(alc.obtenerQ2d(imgPts, objPts))
        w = H @ np.vstack([X, Y, np.ones(77)])
        proj = np.column_stack([w[0] / w[2], w[1] / w[2]])
        err = np.max(np.sqrt(np.sum((proj - imgPts) ** 2, axis=1)))
        assert err < 1.0   # QA previo: 0.37–0.55 px en las 10 imágenes


# ======================================================================
# C — v_ij
# ======================================================================

class TestVij:

    def test_shape(self):
        assert alc.v_ij(np.eye(3), 0, 1).shape == (6,)

    def test_identidad(self):
        """v11(I)=(1,0,0,0,0,0); v12(I)=(0,1,0,0,0,0); v22(I)=(0,0,1,0,0,0)."""
        I = np.eye(3)
        assert np.allclose(alc.v_ij(I, 0, 0), [1, 0, 0, 0, 0, 0], atol=ATOL)
        assert np.allclose(alc.v_ij(I, 0, 1), [0, 1, 0, 0, 0, 0], atol=ATOL)
        assert np.allclose(alc.v_ij(I, 1, 1), [0, 0, 1, 0, 0, 0], atol=ATOL)

    def test_vs_forma_cuadratica(self):
        """LA PRUEBA CLAVE: para H y B simétricas arbitrarias,
        v_ij(H)·b == c_iᵀ B c_j con b = slots de B — en TODO (i,j)."""
        rng = np.random.default_rng(7)
        H = rng.normal(size=(3, 3))
        M = rng.normal(size=(3, 3))
        B = 0.5 * (M + M.T)
        b_exp = _expande_b_desde_B(B)
        for i in range(3):
            for j in range(3):
                cuadratica = H[:, i] @ B @ H[:, j]
                lineal = alc.v_ij(H, i, j) @ b_exp
                assert np.isclose(cuadratica, lineal, atol=1e-9)

    def test_simetria(self):
        H = np.array([[2.0, 1.0, 5.0], [3.0, 7.0, 4.0], [1.0, 1.0, 9.0]])
        for i in range(3):
            for j in range(3):
                assert np.allclose(alc.v_ij(H, i, j), alc.v_ij(H, j, i))


# ======================================================================
# D — armarA
# ======================================================================

class TestArmarA:

    def test_shape(self, datos_sinteticos):
        A = alc.armarA(datos_sinteticos["Hlst"])
        assert A.shape == (2 * len(datos_sinteticos["Hlst"]), 6)

    def test_shape_vacio(self):
        assert alc.armarA([]).shape == (0, 6)

    def test_filas_exactas(self, datos_sinteticos):
        """Cada fila matchea el armado a mano con v_ij (verifica intercalado)."""
        A = alc.armarA(datos_sinteticos["Hlst"])
        for k, H in enumerate(datos_sinteticos["Hlst"]):
            assert np.allclose(A[2*k], alc.v_ij(H, 0, 1), atol=ATOL)
            assert np.allclose(A[2*k+1],
                               alc.v_ij(H, 0, 0) - alc.v_ij(H, 1, 1), atol=ATOL)

    def test_A_b0_con_ground_truth(self, datos_sinteticos):
        """INVARIANTE CLAVE: con las H verdaderas, A·b_true ≈ 0
        (restricciones de la ec. 13 bien armadas)."""
        A = alc.armarA(datos_sinteticos["Hlst"])
        assert np.allclose(A @ datos_sinteticos["b"], 0.0, atol=1e-8)

    def test_H_diagonal(self):
        """H = diag(2,3,1): v12 = (0,6,0,0,0,0) y v11−v22 = (4,0,−9,0,0,0)."""
        A = alc.armarA([np.diag([2.0, 3.0, 1.0])])
        assert np.allclose(A[0], [0, 6, 0, 0, 0, 0], atol=ATOL)
        assert np.allclose(A[1], [4, 0, -9, 0, 0, 0], atol=ATOL)

# ======================================================================
# calcularMatrizK2D
# ======================================================================

class TestCalcularMatrizK2D:

    def test_shape_y_K33(self, datos_sinteticos):
        """K es (3,3) y sale normalizada con K[2,2] = 1."""
        A = alc.armarA(datos_sinteticos["Hlst"])
        K = alc.calcularMatrizK2D(A)
        assert K.shape == (3, 3)
        assert np.isclose(K[2, 2], 1.0, atol=1e-12)

    def test_round_trip_sintetico(self, datos_sinteticos):
        """INVARIANTE CLAVE: K(A de 5 vistas exactas) == K_ground_truth.
        Puramente construido desde K → Hs → proyecciones (no se reimplementa)."""
        d = datos_sinteticos
        K = alc.calcularMatrizK2D(alc.armarA(d["Hlst"]))
        assert np.allclose(K, d["K"], atol=1e-9)

    def test_minimo_3_vistas(self, datos_sinteticos):
        """El enunciado: 3 vistas alcanzan para K completa (γ incluido)."""
        A3 = alc.armarA(datos_sinteticos["Hlst"])[:6]     # 3 vistas = 6 filas
        K = alc.calcularMatrizK2D(A3)
        assert np.allclose(K, datos_sinteticos["K"], atol=1e-9)

    def test_guard_dos_vistas(self, datos_sinteticos):
        """2 vistas (4 filas en A) → AssertionError (< mínimo del enunciado)."""
        A2 = alc.armarA(datos_sinteticos["Hlst"])[:4]
        with pytest.raises(AssertionError):
            alc.calcularMatrizK2D(A2)

    def test_guard_shape_columnas(self, datos_sinteticos):
        """A mal armada (no (·,6)) → AssertionError."""
        A = alc.armarA(datos_sinteticos["Hlst"])
        with pytest.raises(AssertionError):
            alc.calcularMatrizK2D(A[:, :5])

    def test_signo_arbitrario_de_b(self, datos_sinteticos):
        """El sistema homogéneo es mulo el signo: K(−A) == K(A) —
        chequeo directo de la operación "elegir el signo que hace B def. positiva"."""
        A = alc.armarA(datos_sinteticos["Hlst"])
        assert np.allclose(alc.calcularMatrizK2D(-A),
                           alc.calcularMatrizK2D(A), atol=1e-9)

    def test_con_datos_reales_plausibilidad(self):
        """Cadenas completas con los 10 captcha-sheet reales:
        corners → Q → H → A → K. Umbrales de plausibilidad FIJADOS desde la
        corrida 2026-10-27 (no aprobados a mano): fx=742.3, fy=753.6
        (ratio 0.985) y c = (520.5, 299.3), contra el centro teórico del
        formato 1024×576 ~ (512, 288)."""
        archivo = os.path.join(REPO, "calib", "coplanar", "detected_corners.txt")
        if not os.path.exists(archivo):
            pytest.skip("no hay datos reales en esta máquina")
        det = np.loadtxt(archivo)                       # (10, 154)
        objPts = np.column_stack([30.0 * (np.arange(77) % 11),
                                  30.0 * (np.arange(77) // 11),
                                  np.zeros(77)])
        Hs = [alc.resolverDLT2D(alc.obtenerQ2d(det[k].reshape(77, 2), objPts))
              for k in range(det.shape[0])]
        K = alc.calcularMatrizK2D(alc.armarA(Hs))

        assert np.all(np.isfinite(K))                   # sin NaN/inf
        assert K[0, 0] > 0 and K[1, 1] > 0              # focales positivas
        assert abs(K[0, 0] / K[1, 1] - 1.0) < 0.15      # fx ≈ fy (medido: 1.5%)
        assert abs(K[0, 2] - 512.0) < 50.0              # u0 cerca de 512 (medido: +8.5)
        assert abs(K[1, 2] - 288.0) < 50.0              # v0 cerca de 288 (medido: +11.3)

# ======================================================================
# obtenerPose2D
# ======================================================================
class TestObtenerPose2D:

    def test_pose_exacta_sintetica(self, datos_sinteticos):
        """H = K[r1 r2 t] exacta → pose recupera r1, r2, t y R ortogonal."""
        d = datos_sinteticos
        K, H0 = d["K"], d["Hlst"][0]
        X, Y = d["objPts"][:, 0], d["objPts"][:, 1]
        w = H0 @ np.vstack([X, Y, np.ones_like(X)])
        imgPts = np.column_stack([w[0] / w[2], w[1] / w[2]])
        H_rec = alc.resolverDLT2D(alc.obtenerQ2d(imgPts, d["objPts"]))
        H_rec = H_rec / H_rec[2, 2]
        Rj, tj = alc.obtenerPose2D(H_rec, K)

        # R1: (r1, r2, r3) = la parte de rotación de la pose verdadera
        c, s = np.cos(np.deg2rad(20.0)), np.sin(np.deg2rad(20.0))
        r1_true = np.array([c, 0.0, -s])
        r2_true = np.array([0.0, 1.0, 0.0])
        r3_true = np.cross(r1_true, r2_true)
        t_true = np.array([100.0, 50.0, 2000.0]) 
        assert np.allclose(Rj[:, 0], r1_true, atol=1e-6)
        assert np.allclose(Rj[:, 1], r2_true, atol=1e-6)
        assert np.allclose(Rj[:, 2], r3_true, atol=1e-6)
        assert np.allclose(tj, t_true, atol=1e-6)          # t vuelve en unidades de H

    def test_ortogonalidad_y_det(self, datos_sinteticos):
        """Para la vista 0: RᵀR = I y det(R) = +1 con 1e-9."""
        d = datos_sinteticos
        Rj, _ = alc.obtenerPose2D(d["Hlst"][0], d["K"])
        assert np.allclose(Rj.T @ Rj, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(Rj), 1.0, atol=1e-9)

    def test_reproyeccion_pose(self, datos_sinteticos):
        """La pose de vuelta [R|t] reproduce las imágenes con K [R|t]."""
        d = datos_sinteticos
        K, H0, objPts = d["K"], d["Hlst"][0], d["objPts"]
        Rj, tj = alc.obtenerPose2D(H0, K)
        X, Y = objPts[:, 0], objPts[:, 1]
        
        # puntos del plano en 3-D con Su Z=0 (no homogéneos con '1':
        # el r3 de R vive en la dimensión Z del sistema de cámara)
        Z = np.zeros_like(X)
        p_c = Rj @ np.vstack([X, Y, Z]) + tj[:, None]   # K·(r1·x + r2·y + t)
        u = K[0, :] @ p_c / p_c[2]
        v = K[1, :] @ p_c / p_c[2]
        w2 = H0 @ np.vstack([X, Y, np.ones_like(X)])    # vía H: H·[x,y,1] (esto sí lleva '1')
        assert np.allclose(u, w2[0] / w2[2], atol=1e-9)
        assert np.allclose(v, w2[1] / w2[2], atol=1e-9)

    def test_ortogonalidad_todas_las_vistas(self, datos_sinteticos):
        """RRᵀ=I y det=+1 en TODAS las vistas (λ distinto por vista)."""
        d = datos_sinteticos
        for k, H in enumerate(d["Hlst"]):
            Rj, _ = alc.obtenerPose2D(H, d["K"])
            assert np.allclose(Rj.T @ Rj, np.eye(3), atol=1e-9), f"vista {k}"
            assert np.isclose(np.linalg.det(Rj), 1.0, atol=1e-9), f"vista {k}"

    def test_signo_arbitrario_de_H(self, datos_sinteticos):
        """Si el DLT devuelve −H, la pose sigue una rotación válida
        (alternativa R' = R·diag(−1,−1,1), det +1, RRᵀ=I) — el usuario proviene
        del signo arbitrario del vector nulo (mismo fenómeno que el caso de b)."""
        d = datos_sinteticos
        H0 = d["Hlst"][0]
        Rj, _ = alc.obtenerPose2D(-H0, d["K"])
        assert np.allclose(Rj.T @ Rj, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(Rj), 1.0, atol=1e-9)

    def test_con_datos_reales_pose(self):
        """Pose de la imagen 1 real con K de las 10: reproduce las esquinas
        con tolerancia holgada (el ruido de H rompe la ortogonalidad exacta
        — la banda de power de este test guard es Hartley, issue #8)."""
        archivo = os.path.join(REPO, "calib", "coplanar", "detected_corners.txt")
        if not os.path.exists(archivo):
            pytest.skip("no hay datos reales en esta máquina")
        det = np.loadtxt(archivo)
        objPts = np.column_stack([30.0*(np.arange(77)%11),
                                  30.0*(np.arange(77)//11),
                                  np.zeros(77)])
        Hs = [alc.resolverDLT2D(alc.obtenerQ2d(det[k].reshape(77,2), objPts))
              for k in range(det.shape[0])]
        K = alc.calcularMatrizK2D(alc.armarA(Hs))
        Rj, tj = alc.obtenerPose2D(Hs[0], K)

        # ortogonalidad "aproximada" del nivel del ruido pixel (medida al volar)
        assert np.linalg.norm(Rj.T @ Rj - np.eye(3)) < 0.05

        # reprojection path de la pose
        imgPts = det[0].reshape(77, 2)
        X, Y = objPts[:, 0], objPts[:, 1]
        p_c = Rj @ np.vstack([X, Y, np.zeros(77)]) + tj[:, None]
        u = K[0, :] @ p_c / p_c[2]
        v = K[1, :] @ p_c / p_c[2]
        err = np.max(np.sqrt((u - imgPts[:, 0])**2 + (v - imgPts[:, 1])**2))
        assert err < 5.0