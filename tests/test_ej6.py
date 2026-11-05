"""Tests del Ejercicio 6 — POSIT coplanar (Aterrizaje suave).

Los tests no forman parte del código entregado: acá se usa np.linalg, @ y
np.cross como referencia independiente.

Correr desde la raíz del repo:  python -m pytest tests/test_ej6.py -v
"""

import os
import sys

import numpy as np
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
import alc  # noqa: E402

# Puntos de la H (enunciado / heli/object_points.csv), orden M0..M4
PUNTOS_H = np.array([[0.000, 0.0, 0.0],
                     [-0.175, 0.3, 0.0],
                     [0.175, 0.3, 0.0],
                     [0.175, -0.3, 0.0],
                     [-0.175, -0.3, 0.0]])
K_HELI = np.loadtxt(os.path.join(RAIZ, "heli", "K.csv"), delimiter=";")


def rotacion(alfa, beta, gama):
    """Rz(gama)·Ry(beta)·Rx(alfa) — construcción independiente de R."""
    ca, sa, cb, sb, cg, sg = (np.cos(alfa), np.sin(alfa), np.cos(beta),
                              np.sin(beta), np.cos(gama), np.sin(gama))
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def proyectar(R, t, puntos, K):
    """Proyección pinhole de referencia: (u, v) = π(K·(R·X + t))."""
    homogeneos = (puntos @ R.T + t) @ K.T
    return homogeneos[:, :2] / homogeneos[:, 2:]


def es_rotacion(R, tol=1e-12):
    return np.allclose(R.T @ R, np.eye(3), atol=tol) and abs(np.linalg.det(R) - 1) < tol


# Poses sintéticas con la H vista desde arriba (eje óptico ≈ -W), inclinación
# >= 20° y Z >= 2 m, región donde la iteración de POSIT converge. La pose 2
# es la misma rotación del caso a 1.5 m, que no converge por el método y
# queda en test_limitacion_cuenca_de_convergencia.
POSES_SINTETICAS = [
    (rotacion(np.pi - 0.45, 0.10, 1.40), np.array([0.30, -0.20, 4.0])),
    (rotacion(np.pi + 0.20, -0.40, 0.30), np.array([-0.50, 0.40, 2.5])),
    (rotacion(np.pi + 0.60, 0.30, -2.00), np.array([0.10, 0.05, 7.0])),
]


# ----------------------------------------------------------------------
# 1. Caso sintético con R, t conocidos (y fx != fy → requisito 4)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("R_real, t_real", POSES_SINTETICAS)
def test_posit_recupera_pose_sintetica(R_real, t_real):
    pixeles = proyectar(R_real, t_real, PUNTOS_H, K_HELI)
    R, t = alc.aplicarPOSIT(PUNTOS_H, pixeles, K_HELI, 500, 1e-13)
    # Datos exactos: la tolerancia del README (1e-9) aplica a R y t
    assert np.allclose(R, R_real, atol=1e-9)
    assert np.allclose(t, t_real, atol=1e-9)


def test_posit_con_m0_fuera_del_origen():
    # Se traslada el objeto: M0 = (0.2, -0.1, 0.5). t debe absorber R·M0.
    desplazamiento = np.array([0.2, -0.1, 0.5])
    puntos = PUNTOS_H + desplazamiento
    R_real, t_real = POSES_SINTETICAS[0]
    pixeles = proyectar(R_real, t_real, puntos, K_HELI)
    R, t = alc.aplicarPOSIT(puntos, pixeles, K_HELI, 500, 1e-13)
    assert np.allclose(R, R_real, atol=1e-9) and np.allclose(t, t_real, atol=1e-9)


# ----------------------------------------------------------------------
# 2. λ y μ: dos soluciones y restricciones
# ----------------------------------------------------------------------
@pytest.mark.parametrize("semilla", range(5))
def test_lambda_mu_cumple_restricciones(semilla):
    rng = np.random.default_rng(semilla)
    I0 = np.hstack([rng.normal(size=2) * 300, 0.0])
    J0 = np.hstack([rng.normal(size=2) * 300, 0.0])
    (l1, m1), (l2, m2) = alc.calculoLambdaMu(I0, J0)
    assert (l2, m2) == (-l1, -m1) and l1 >= 0
    escala = np.dot(I0, I0) + np.dot(J0, J0)
    for lam, mu in [(l1, m1), (l2, m2)]:
        assert abs(lam * mu + np.dot(I0, J0)) < 1e-12 * escala
        assert abs(lam**2 - mu**2 - (np.dot(J0, J0) - np.dot(I0, I0))) < 1e-12 * escala
        # Consecuencia: I y J ortogonales y de igual norma
        I, J = I0 + lam * np.array([0, 0, 1]), J0 + mu * np.array([0, 0, 1])
        assert abs(np.dot(I, J)) < 1e-12 * escala
        assert abs(np.dot(I, I) - np.dot(J, J)) < 1e-12 * escala


def test_lambda_mu_casos_degenerados():
    # I0, J0 ya ortogonales y de igual norma → λ = μ = 0
    assert alc.calculoLambdaMu(np.array([2.0, 0, 0]), np.array([0, 2.0, 0])) == ((0, 0), (0, 0))
    # I0·J0 = 0 y ‖I0‖ > ‖J0‖ → λ = 0, μ = ±√(‖I0‖² - ‖J0‖²)
    (l1, m1), _ = alc.calculoLambdaMu(np.array([5.0, 0, 0]), np.array([0, 3.0, 0]))
    assert l1 == 0 and abs(abs(m1) - 4) < 1e-12


# ----------------------------------------------------------------------
# 3. Reconstrucción de rotación válida
# ----------------------------------------------------------------------
@pytest.mark.parametrize("R_real, t_real", POSES_SINTETICAS)
def test_salida_posit_desde_I_J_exactos(R_real, t_real):
    # I = (f/Z0)·i, J = (f/Z0)·j y m0 = imagen virtual de M0 (aquí M0 = 0)
    fx = 1500.0
    Z0 = t_real[2]
    m0 = fx * t_real[:2] / Z0
    salida = alc.calcularSalidaPOSIT(fx / Z0 * R_real[0], fx / Z0 * R_real[1], fx, m0, np.zeros(3))
    assert len(salida) == 6
    eje_i, eje_j, eje_k, Z0_calculado, R, t = salida
    assert es_rotacion(R)
    assert np.allclose(R, R_real, atol=1e-12) and np.allclose(t, t_real, atol=1e-12)
    # i, j, k son las filas de R y Z0 la profundidad de M0
    assert np.array_equal(np.vstack([eje_i, eje_j, eje_k]), R)
    assert np.allclose(eje_k, np.cross(eje_i, eje_j), atol=1e-15)
    assert abs(Z0_calculado - Z0) < 1e-12


# ----------------------------------------------------------------------
# 4. Reproyección con fx != fy
# ----------------------------------------------------------------------
def test_evaluar_pose_con_fx_distinto_fy():
    assert K_HELI[0, 0] != K_HELI[1, 1]
    R, t = POSES_SINTETICAS[0]
    pixeles = proyectar(R, t, PUNTOS_H, K_HELI)
    salida = alc.evaluarPose(R, t, PUNTOS_H, pixeles, K_HELI)
    assert len(salida) == 2
    n_validos, rms = salida
    assert rms < 1e-9 and n_validos == 5
    proyectados, profundidades = alc.proyectarPuntos(R, t, PUNTOS_H, K_HELI)
    assert np.allclose(proyectados, pixeles, atol=1e-9)
    assert np.allclose(profundidades, (PUNTOS_H @ R.T + t)[:, 2], atol=1e-15)
    # Si se usara fy = fx la reproyección sería otra: el RMS lo detecta
    K_igual = K_HELI.copy()
    K_igual[1, 1] = K_igual[0, 0]
    assert alc.evaluarPose(R, t, PUNTOS_H, pixeles, K_igual)[1] > 1.0


# ----------------------------------------------------------------------
# 5. Descarte de poses con profundidades negativas
# ----------------------------------------------------------------------
def test_evaluar_pose_detras_de_la_camara():
    R, t = POSES_SINTETICAS[0]
    pixeles = proyectar(R, t, PUNTOS_H, K_HELI)
    n_validos, rms = alc.evaluarPose(R, -t, PUNTOS_H, pixeles, K_HELI)
    assert rms == np.inf and n_validos == 0
    assert np.all(np.isnan(alc.proyectarPuntos(R, -t, PUNTOS_H, K_HELI)[0]))
    # Solo un punto detrás: el resto se reproyecta pero el RMS sigue inválido
    R_parcial, t_parcial = np.diag([1.0, -1, -1]) @ rotacion(0.5, 0, 0), np.array([0.0, 0.0, 0.05])
    n_validos, rms = alc.evaluarPose(R_parcial, t_parcial, PUNTOS_H, pixeles, K_HELI)
    assert rms == np.inf and 0 < n_validos < 5
    proyectados, _ = alc.proyectarPuntos(R_parcial, t_parcial, PUNTOS_H, K_HELI)
    assert np.isnan(proyectados).any() and not np.isnan(proyectados).all()


# ----------------------------------------------------------------------
# 6. MaxIt y atol
# ----------------------------------------------------------------------
def test_maxit_limita_iteraciones():
    pixeles = cargar_pixeles_medidos()
    _, _, info = alc.iterarPOSIT(PUNTOS_H, pixeles, K_HELI, 1, 0.0)
    assert info['iteraciones'] == 1 and info['motivo'] == 'MaxIt'
    assert len(info['historial_rms']) == 1
    _, _, info = alc.iterarPOSIT(PUNTOS_H, pixeles, K_HELI, 3, 0.0)
    assert info['iteraciones'] == 3 and info['motivo'] == 'MaxIt'


@pytest.mark.parametrize("atol", [0.0, 1e-6, 0.1, 0.5, 1e6])
def test_criterio_de_parada_literal(atol):
    """Inciso e: se itera mientras RMS_anterior - RMS_actual >= atol y se
    corta en la primera iteración en que no se cumple (sin valor absoluto:
    un aumento del RMS también corta). Con atol enorme corta en la 2."""
    _, _, info = alc.iterarPOSIT(PUNTOS_H, cargar_pixeles_medidos(), K_HELI, 100, atol)
    mejoras = -np.diff(info['historial_rms'])
    assert info['motivo'] == 'atol'
    assert np.all(mejoras[:-1] >= atol) and mejoras[-1] < atol


def test_devuelve_la_mejor_pose_del_historial():
    pixeles = cargar_pixeles_medidos()
    R, t, info = alc.iterarPOSIT(PUNTOS_H, pixeles, K_HELI, 100, 0.0)
    # Datos reales: la última iteración empeora el RMS, la devuelta es la anterior
    assert info['historial_rms'][-1] > info['rms'] == min(info['historial_rms'])
    assert info['rms'] == alc.evaluarPose(R, t, PUNTOS_H, pixeles, K_HELI)[1]


def test_aplicar_posit_devuelve_solo_R_t():
    pixeles = cargar_pixeles_medidos()
    salida = alc.aplicarPOSIT(PUNTOS_H, pixeles, K_HELI, 100, 1e-6)
    assert len(salida) == 2
    R, t = salida
    assert R.shape == (3, 3) and t.shape == (3,)
    R_diag, t_diag, _ = alc.iterarPOSIT(PUNTOS_H, pixeles, K_HELI, 100, 1e-6)
    assert np.array_equal(R, R_diag) and np.array_equal(t, t_diag)


def test_entradas_invalidas():
    pixeles = proyectar(*POSES_SINTETICAS[0], PUNTOS_H, K_HELI)
    with pytest.raises(AssertionError):      # cantidades distintas
        alc.aplicarPOSIT(PUNTOS_H, pixeles[:4], K_HELI, 10, 1e-6)
    with pytest.raises(AssertionError):      # puntos colineales
        colineales = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0.0]])
        alc.aplicarPOSIT(colineales, pixeles, K_HELI, 10, 1e-6)
    with pytest.raises(AssertionError):      # no coplanares en W
        alc.aplicarPOSIT(PUNTOS_H + np.array([0, 0, 0.1]) * np.arange(5)[:, None],
                         pixeles, K_HELI, 10, 1e-6)
    with pytest.raises(AssertionError):
        alc.aplicarPOSIT(PUNTOS_H, pixeles, K_HELI, 0, 1e-6)


@pytest.mark.parametrize("R_real, t_real", [
    (rotacion(np.pi + 0.05, 0.03, 0.5), np.array([0.1, 0.1, 1.5])),    # casi frontoparalela
    (rotacion(np.pi + 0.20, -0.40, 0.30), np.array([-0.5, 0.4, 1.5])),  # 25°, muy cerca
])
@pytest.mark.xfail(strict=True, reason="limitación del método: la pose real no es "
                   "atractor de la iteración de POSIT (radio espectral >= 0.995)")
def test_limitacion_cuenca_de_convergencia(R_real, t_real):
    """Documenta la limitación del método (no de la implementación): la pose
    real es punto fijo exacto de ε → ε', pero el radio espectral de ese mapa
    en la pose real es 1.63 (casi frontoparalela, repulsor) y 0.995 (25° a
    1.5 m, prácticamente neutro), así que la iteración no llega a ella.
    Medido con un jacobiano numérico durante el desarrollo; en cambio a
    Z = 2.5 m el radio es 0.60 y converge (POSES_SINTETICAS[1]).
    Se exige la pose real y se marca xfail estricto: si algún día pasa,
    pytest lo informa como error para revisar esta documentación."""
    pixeles = proyectar(R_real, t_real, PUNTOS_H, K_HELI)
    R, t = alc.aplicarPOSIT(PUNTOS_H, pixeles, K_HELI, 500, 0.0)
    # Aun sin converger, la salida es una pose válida (R ortonormal, Z > 0)
    assert es_rotacion(R) and np.all((PUNTOS_H @ R.T + t)[:, 2] > 0), "pose inválida"
    assert np.allclose(R, R_real, atol=1e-6) and np.allclose(t, t_real, atol=1e-6)


# ----------------------------------------------------------------------
# 7. Datos reales: row_00248.png con los cinco puntos medidos a mano
# ----------------------------------------------------------------------
def cargar_pixeles_medidos():
    return np.loadtxt(os.path.join(RAIZ, "scripts", "puntos_imagen_ej6.csv"),
                      delimiter=";", skiprows=1, usecols=(1, 2))


def test_csv_de_objeto_coincide_con_enunciado():
    puntos = np.loadtxt(os.path.join(RAIZ, "heli", "object_points.csv"),
                        delimiter=";", skiprows=1, usecols=(1, 2, 3))
    assert np.array_equal(puntos, PUNTOS_H)


def test_posit_datos_reales():
    """Regresión con los puntos medidos (README, nivel 3). Durante el
    desarrollo la pose se contrastó con un refinamiento no lineal del error
    de reproyección (óptimo 0.88 px, a ≈ 9 cm en C y < 3° en R)."""
    pixeles = cargar_pixeles_medidos()
    R, t = alc.aplicarPOSIT(PUNTOS_H, pixeles, K_HELI, 100, 1e-6)
    n_validos, rms = alc.evaluarPose(R, t, PUNTOS_H, pixeles, K_HELI)
    centro = -R.T @ t
    assert es_rotacion(R)
    # Clicks manuales con precisión ~1 px: se exige RMS < 2 px
    assert n_validos == 5 and rms < 2.0
    # La cámara está por encima del suelo (W > 0) y mira hacia abajo (k_W < 0)
    assert centro[2] > 0 and R[2, 2] < 0
    # Baseline: RMS = 1.083 px y C = (-0.647, -0.732, 4.495) m
    assert abs(rms - 1.0828) < 1e-3
    assert np.allclose(centro, [-0.6474, -0.7321, 4.4952], atol=1e-3)


# ----------------------------------------------------------------------
# 8. Inciso f: pose objetivo
# ----------------------------------------------------------------------
@pytest.mark.parametrize("h", [0.5, 2.0, 4.5])
def test_pose_objetivo(h):
    R, t = alc.calcularPoseObjetivo(h)
    assert es_rotacion(R)
    # Centro óptico sobre M0 a altura h y eje óptico hacia el suelo (-W)
    assert np.allclose(-R.T @ t, [0, 0, h]) and np.allclose(R[2], [0, 0, -1])
    # Toda la H delante de la cámara; M0 cae en el punto principal
    assert np.all((PUNTOS_H @ R.T + t)[:, 2] > 0)
    pixeles = proyectar(R, t, PUNTOS_H, K_HELI)
    assert np.allclose(pixeles[0], K_HELI[:2, 2])
    # H derecha: M1 (sup. izq., punto blanco) arriba a la izquierda, M3 abajo a la derecha
    cx, cy = K_HELI[0, 2], K_HELI[1, 2]
    assert pixeles[1, 0] < cx and pixeles[1, 1] < cy
    assert pixeles[2, 0] > cx and pixeles[2, 1] < cy
    assert pixeles[3, 0] > cx and pixeles[3, 1] > cy


def test_pose_objetivo_altura_invalida():
    with pytest.raises(AssertionError):
        alc.calcularPoseObjetivo(0.0)
