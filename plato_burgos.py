"""Plato con el relieve real de la provincia de Burgos y hueco central para una taza.

  - Contorno: spain-provinces.geojson (click_that_hood, GitHub).
  - Relieve: AWS Terrain Tiles (formato Terrarium), zoom 10 (~110 m/pixel).
  - Plato de 210 mm con borde; la provincia centrada en Burgos capital, exagerada en vertical.
  - Hueco de 61 mm (taza de 60 mm + 1 mm de holgura) en Burgos capital, centro del plato.
Salida: plato_burgos.stl y plato_burgos_visor.html (visor 3D, doble clic)
"""
import io
import json
import math
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
import manifold3d as mf
import trimesh

AQUI = Path(__file__).resolve().parent
DATOS = AQUI / 'datos'
DATOS.mkdir(exist_ok=True)

# ---------------- parametros ----------------
D_PLATO = 210.0          # diametro del plato
GROSOR = 3.0             # grosor del fondo del plato
BORDE_ALTO = 6.0         # altura del borde
BORDE_ANCHO = 8.0
R_UTIL = 94.0            # radio maximo que puede ocupar la provincia
Z_BASE = GROSOR + 1.2    # cota del relieve en la cota mas baja de la provincia
ALTO_RELIEVE = 12.0      # diferencia de altura entre el punto mas bajo y el mas alto
D_TAZA = 60.0 + 1.0
FONDO_HUECO = 2.0        # cota del fondo del hueco (1 mm dentro del plato)
BURGOS = (-3.6969, 42.3439)   # Burgos capital (lon, lat) = centro del plato
ZOOM = 10

# ---------------- contorno ----------------
gj = json.loads((DATOS / 'spain-provinces.geojson').read_text(encoding='utf-8'))
feat = next(f for f in gj['features'] if f['properties'].get('name') == 'Burgos')
g = feat['geometry']
anillos = g['coordinates'] if g['type'] == 'Polygon' else [p[0] for p in g['coordinates']]
if g['type'] == 'Polygon':
    anillos = [g['coordinates'][0]]
anillos = [np.array(a, dtype=float) for a in anillos]
print('partes del contorno:', len(anillos), [len(a) for a in anillos])

lon0, lat0 = BURGOS
KX = 111.32 * math.cos(math.radians(lat0))   # km por grado de longitud
KY = 110.57                                  # km por grado de latitud
todos = np.vstack(anillos)
dist_max = np.max(np.hypot((todos[:, 0] - lon0) * KX, (todos[:, 1] - lat0) * KY))
ESC = R_UTIL / dist_max                      # mm por km
print(f'escala: {ESC:.3f} mm/km  (1:{1e6 / ESC:,.0f})')


def a_mm(lon, lat):
    return (lon - lon0) * KX * ESC, (lat - lat0) * KY * ESC


def a_lonlat(x, y):
    return lon0 + x / (KX * ESC), lat0 + y / (KY * ESC)


# ---------------- relieve (teselas Terrarium) ----------------
def tesela_xy(lon, lat, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


lon_min, lat_min = todos.min(axis=0)
lon_max, lat_max = todos.max(axis=0)
x0, y0 = [int(v) for v in tesela_xy(lon_min, lat_max, ZOOM)]
x1, y1 = [int(v) for v in tesela_xy(lon_max, lat_min, ZOOM)]
print(f'teselas: {(x1 - x0 + 1) * (y1 - y0 + 1)}')
mosaico = np.zeros(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256), dtype=np.float32)
for tx in range(x0, x1 + 1):
    for ty in range(y0, y1 + 1):
        f = DATOS / f'terrarium_{ZOOM}_{tx}_{ty}.png'
        if not f.exists():
            url = f'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{ZOOM}/{tx}/{ty}.png'
            f.write_bytes(urllib.request.urlopen(url, timeout=60).read())
        im = np.asarray(Image.open(f).convert('RGB')).astype(np.float32)
        elev = im[..., 0] * 256 + im[..., 1] + im[..., 2] / 256 - 32768
        mosaico[(ty - y0) * 256:(ty - y0 + 1) * 256, (tx - x0) * 256:(tx - x0 + 1) * 256] = elev
mosaico = gaussian_filter(mosaico, 2.2)        # suaviza aristas duras del terreno


def elevacion(lon, lat):
    n = 2 ** ZOOM
    px = ((lon + 180) / 360 * n - x0) * 256
    py = ((1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n - y0) * 256
    return map_coordinates(mosaico, [py - .5, px - .5], order=1, mode='nearest')


# rango de alturas dentro de la provincia (muestreo de los vertices + rejilla)
def chaikin(p, vueltas=3):
    """Suaviza un anillo cerrado (corta las esquinas) para un contorno organico."""
    for _ in range(vueltas):
        q = np.roll(p, -1, axis=0)
        p = np.vstack([np.column_stack([.75 * p[:, 0] + .25 * q[:, 0], .75 * p[:, 1] + .25 * q[:, 1]]),
                       np.column_stack([.25 * p[:, 0] + .75 * q[:, 0], .25 * p[:, 1] + .75 * q[:, 1]])])
        p = p.reshape(2, -1, 2).transpose(1, 0, 2).reshape(-1, 2)
    return p


def area(p):
    return .5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], 1)) - np.dot(p[:, 1], np.roll(p[:, 0], 1)))


mm_anillos = [np.column_stack(a_mm(a[:, 0], a[:, 1])) for a in anillos]
mm_anillos = [chaikin(p) for p in mm_anillos if area(p) > 6]     # fuera restos diminutos
seccion = mf.CrossSection(mm_anillos, mf.FillRule.EvenOdd)

# distancia al borde (para bajar el relieve en rampa hacia el plato)
from scipy.spatial import cKDTree
puntos_borde = []
for p in mm_anillos:
    q = np.roll(p, -1, axis=0)
    for (x1, y1), (x2, y2) in zip(p, q):
        n = max(1, int(math.hypot(x2 - x1, y2 - y1) / .25))
        t = np.arange(n) / n
        puntos_borde.append(np.column_stack([x1 + (x2 - x1) * t, y1 + (y2 - y1) * t]))
arbol_borde = cKDTree(np.vstack(puntos_borde))
gx, gy = np.meshgrid(np.arange(-R_UTIL, R_UTIL, 1.0), np.arange(-R_UTIL, R_UTIL, 1.0))
lo, la = a_lonlat(gx.ravel(), gy.ravel())
from matplotlib.path import Path as MPath
dentro = np.zeros(gx.size, bool)
for p in mm_anillos:
    dentro ^= MPath(p).contains_points(np.column_stack([gx.ravel(), gy.ravel()]))
e_in = elevacion(lo[dentro], la[dentro])
E_MIN, E_MAX = float(np.percentile(e_in, .5)), float(e_in.max())
print(f'alturas en la provincia: {E_MIN:.0f} - {E_MAX:.0f} m  (exageracion vertical x{ALTO_RELIEVE / ((E_MAX - E_MIN) / 1000 * ESC):.1f})')


def cota(x, y):
    lon, lat = a_lonlat(x, y)
    e = np.clip(elevacion(lon, lat), E_MIN, None)
    return Z_BASE + (e - E_MIN) / (E_MAX - E_MIN) * ALTO_RELIEVE


# ---------------- solidos ----------------
Z_PIE = GROSOR - .5                          # el relieve arranca dentro del plato (se fusionan)
H = 1.0
relieve = mf.Manifold.extrude(seccion, H).translate((0, 0, Z_PIE)).refine_to_length(.6)


RAMPA = 6.0                                  # mm en los que el relieve baja hasta el plato
Z_BORDE = GROSOR + .4                        # altura del relieve justo en el contorno


def cota_relieve(x, y):
    """Cota con el borde en rampa suave (sin escalon vertical en el contorno)."""
    d, _ = arbol_borde.query(np.column_stack([x, y]))
    s = np.clip(d / RAMPA, 0, 1)
    w = s * s * (3 - 2 * s)                  # smoothstep
    return Z_BORDE + (cota(x, y) - Z_BORDE) * w


def deforma(v):
    v = np.array(v, dtype=np.float64)
    t = (v[:, 2] - Z_PIE) / H
    v[:, 2] = Z_PIE + t * (cota_relieve(v[:, 0], v[:, 1]) - Z_PIE)
    return v


relieve = relieve.warp_batch(deforma)

R = D_PLATO / 2
perfil = mf.CrossSection([np.array([
    (0, 0), (R, 0), (R, BORDE_ALTO - .8), (R - .8, BORDE_ALTO), (R - 2.2, BORDE_ALTO),
    (R - BORDE_ANCHO, GROSOR), (0, GROSOR)])])
plato = mf.Manifold.revolve(perfil, 360)

def revoluciona(perfil, x, y):
    """Solido de revolucion a partir de un perfil (r, z), colocado en (x, y)."""
    return mf.Manifold.revolve(mf.CrossSection([np.array(perfil, dtype=float)]), 160).translate((x, y, 0))


def cota_en(x, y):
    return float(cota(np.array([x]), np.array([y]))[0])


# --- Cuenco de Aranda de Duero: crater liso con labio ---
ax, ay = a_mm(-3.6887, 41.6705)
zt = cota_en(ax, ay)
labio = zt + 2.5
# falda exterior en rampa suave desde el plato hasta el labio (sin pared vertical)
falda = [(23.5, Z_BORDE - .1)]
for k in range(1, 9):
    t = k / 8
    falda.append((23.5 - 7 * t, Z_BORDE - .1 + (labio - .3 - Z_BORDE + .1) * (t * t * (3 - 2 * t))))
crater = revoluciona([(0, Z_PIE), (23.5, Z_PIE)] + falda + [(15.6, labio - .1), (15.0, labio),
                      (13.8, labio), (0, labio)], ax, ay)
a, fondo = 13.5, GROSOR - .5                     # radio de la boca y cota del fondo
d = labio - fondo
r_esf = (a * a + d * d) / (2 * d)
cuenco = (mf.Manifold.sphere(r_esf, 200).translate((ax, ay, fondo + r_esf))
          ^ mf.Manifold.cylinder(80, a + .01, a + .01, 160).translate((ax, ay, fondo - 1)))

# --- Monticulo de Miranda de Ebro: cerro natural (meseta, laderas con crestas y barrancos) ---
mx, my = a_mm(-2.9469, 42.6865)
CIMA = 18.0
R_M = 19.0


def radio_m(th):                              # planta irregular
    return R_M * (1 + .07 * np.sin(3 * th + 1.1) + .05 * np.sin(5 * th + 2.3) + .03 * np.sin(9 * th + .4))


def cota_monticulo(x, y):
    dx, dy = x - mx, y - my
    th = np.arctan2(dy, dx)
    s = np.hypot(dx, dy) / radio_m(th)        # 0 en el centro, 1 en el pie
    sc = np.clip(s, 0, 1)
    meseta = .34
    u = np.clip((sc - meseta) / (1 - meseta), 0, 1)
    ladera = (1 - u) ** 1.7 * (1 - .15 * u)   # ladera concava
    # crestas y barrancos (mas marcados a media ladera) + rugosidad
    ondas = (.55 * np.sin(7 * th + 4 * sc) + .3 * np.sin(13 * th - 6 * sc + 1) + .15 * np.sin(23 * th + 2))
    surcos = np.sin(np.pi * u) * ondas * 1.3
    cumbre = -.9 * (sc / meseta) ** 2 * (sc < meseta) - .9 * (sc >= meseta) + .25 * np.sin(5 * th) * (sc < meseta)
    h = Z_BORDE + (CIMA - Z_BORDE) * ladera + surcos * (sc > meseta) + cumbre + .9
    h = np.where(sc >= 1, Z_BORDE - .3, h)
    return np.minimum(h, CIMA)


th_m = np.linspace(0, 2 * np.pi, 240, endpoint=False)
planta_m = np.column_stack([mx + radio_m(th_m) * np.cos(th_m), my + radio_m(th_m) * np.sin(th_m)])
monticulo = (mf.Manifold.extrude(mf.CrossSection([planta_m]), H).translate((0, 0, Z_PIE))
             .refine_to_length(.45))


def deforma_m(v):
    v = np.array(v, dtype=np.float64)
    t = (v[:, 2] - Z_PIE) / H
    v[:, 2] = Z_PIE + t * (cota_monticulo(v[:, 0], v[:, 1]) - Z_PIE)
    return v


monticulo = monticulo.warp_batch(deforma_m)

hueco = mf.Manifold.cylinder(60, D_TAZA / 2, D_TAZA / 2, 256).translate((0, 0, FONDO_HUECO))
pieza = (plato + relieve + crater + monticulo) - hueco - cuenco
print(f'cuenco en Aranda ({ax:.1f}, {ay:.1f}) prof {labio - fondo:.1f} mm | monticulo en Miranda ({mx:.1f}, {my:.1f})')
print('estado:', pieza.status(), '| triangulos:', pieza.num_tri(), '| volumen cm3:', round(pieza.volume() / 1000, 1))

m = pieza.to_mesh()
malla = trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=False)
print('cerrada (watertight):', malla.is_watertight, '| caja:', np.round(malla.bounds, 1).tolist())
malla.export(AQUI / 'plato_burgos.stl')
print('STL guardado')

# visor 3D con el STL incrustado (plantilla visor.html -> plato_burgos_visor.html)
import base64
html = (AQUI / 'visor.html').read_text(encoding='utf-8')
b64 = base64.b64encode((AQUI / 'plato_burgos.stl').read_bytes()).decode()
(AQUI / 'plato_burgos_visor.html').write_text(html.replace('/*STL*/', b64), encoding='utf-8')
(AQUI / 'index.html').write_text(html.replace('/*STL*/', b64), encoding='utf-8')   # portada de GitHub Pages
print('visor guardado')

