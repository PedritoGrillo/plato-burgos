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
mosaico = gaussian_filter(mosaico, 1.5)


def elevacion(lon, lat):
    n = 2 ** ZOOM
    px = ((lon + 180) / 360 * n - x0) * 256
    py = ((1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n - y0) * 256
    return map_coordinates(mosaico, [py - .5, px - .5], order=1, mode='nearest')


# rango de alturas dentro de la provincia (muestreo de los vertices + rejilla)
mm_anillos = [np.column_stack(a_mm(a[:, 0], a[:, 1])) for a in anillos]
seccion = mf.CrossSection(mm_anillos, mf.FillRule.EvenOdd)
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


def deforma(v):
    v = np.array(v, dtype=np.float64)
    t = (v[:, 2] - Z_PIE) / H
    v[:, 2] = Z_PIE + t * (cota(v[:, 0], v[:, 1]) - Z_PIE)
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
crater = revoluciona([(0, Z_PIE), (20, Z_PIE), (20, zt - .6), (16.5, labio - .3), (15.2, labio),
                      (13.8, labio), (0, labio)], ax, ay)
a, fondo = 13.5, GROSOR - .5                     # radio de la boca y cota del fondo
d = labio - fondo
r_esf = (a * a + d * d) / (2 * d)
cuenco = (mf.Manifold.sphere(r_esf, 200).translate((ax, ay, fondo + r_esf))
          ^ mf.Manifold.cylinder(80, a + .01, a + .01, 160).translate((ax, ay, fondo - 1)))

# --- Monticulo de Miranda de Ebro: peana con falda curva y cima plana ---
mx, my = a_mm(-2.9469, 42.6865)
CIMA = 18.0
perfil = [(0, Z_PIE)]
for k in range(21):
    t = k / 20
    perfil.append((10 + 7 * (1 - t) ** 2.2, Z_PIE + t * (CIMA - .6 - Z_PIE)))
perfil += [(9.6, CIMA - .15), (9.0, CIMA), (0, CIMA)]
monticulo = revoluciona(perfil, mx, my)

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

