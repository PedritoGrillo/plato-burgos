# Plato con el relieve de Burgos

Plato de 210 mm con el relieve real de la provincia de Burgos y un hueco central de 61 mm para una taza,
pensado para imprimir en 3D (FDM, sin soportes).

**Visor 3D:** https://pedritogrillo.github.io/plato-burgos/

- `plato_burgos.stl`: modelo para imprimir.
- `plato_burgos.py`: generador. Descarga el relieve y el contorno, recorta la provincia, exagera la altura (x8)
  y construye el sólido con manifold3d. También crea el visor (`index.html`) con el STL incrustado.

Datos: AWS Terrain Tiles (Mapzen; SRTM y otros) y contorno de provincias de
[click_that_hood](https://github.com/codeforgermany/click_that_hood).
