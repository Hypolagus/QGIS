# -*- coding: utf-8 -*-
"""
Generate XYZ Tiles with Metatile Buffer (Zoptymalizowany dla WebP / JPG / PNG)
=============================================================================
"""

import os
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from qgis.PyQt.QtCore import QSize, QEventLoop, QTimer
from qgis.PyQt.QtGui import QColor, QImage
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterExtent,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterColor,
    QgsMapSettings,
    QgsMapRendererParallelJob,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsRectangle,
    QgsProject,
    QgsPointXY,
    QgsFeatureRequest,
)


class Tile:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def to_point_wgs84(self, n=None):
        if n is None:
            n = 2.0 ** self.z
        lon = self.x / n * 360.0 - 180.0
        lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * self.y / n))))
        return QgsPointXY(lon, lat)


def tiles_for_extent(extent_wgs84, zoom_min, zoom_max):
    result = {}
    for z in range(zoom_min, zoom_max + 1):
        n = 2 ** z
        x_min = int((extent_wgs84.xMinimum() + 180.0) / 360.0 * n)
        x_max = int((extent_wgs84.xMaximum() + 180.0) / 360.0 * n)

        lat_min_rad = math.radians(extent_wgs84.yMinimum())
        lat_max_rad = math.radians(extent_wgs84.yMaximum())
        y_min = int((1.0 - math.log(math.tan(lat_max_rad) + 1.0 / math.cos(lat_max_rad)) / math.pi) / 2.0 * n)
        y_max = int((1.0 - math.log(math.tan(lat_min_rad) + 1.0 / math.cos(lat_min_rad)) / math.pi) / 2.0 * n)

        tiles = [
            (x, y)
            for x in range(x_min, x_max + 1)
            for y in range(y_min, y_max + 1)
        ]
        result[z] = tiles
    return result


class MetatileRenderer:
    TILE_W = 256
    TILE_H = 256

    def __init__(self, output_dir, extent_3857, dpi=96, background=None,
                 antialias=True, img_format='WEBP', quality=90,
                 metatile_size=4, use_buffer=True, feedback=None,
                 skip_empty=False, buffer_tiles=1):

        self.output_dir = output_dir
        self.extent_3857 = extent_3857
        self.dpi = dpi
        self.background = background or QColor(255, 255, 255, 0)
        self.antialias = antialias
        self.img_format = img_format.upper()
        self.quality = quality
        self.metatile_size = metatile_size
        self.use_buffer = use_buffer
        self.buffer_tiles = max(0, int(buffer_tiles))
        self.feedback = feedback
        self.skip_empty = skip_empty
        self._feat_index = None

        self.ext = {
            'PNG': 'png',
            'JPEG': 'jpg',
            'WEBP': 'webp',
        }.get(self.img_format, 'webp')

        self._created_dirs = set()
        self.io_pool = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

        self.map_settings = QgsMapSettings()
        self.map_settings.setBackgroundColor(self.background)
        self.map_settings.setOutputDpi(dpi)
        self.map_settings.setDestinationCrs(QgsCoordinateReferenceSystem('EPSG:3857'))
        self.map_settings.setFlag(QgsMapSettings.Flag.UseAdvancedEffects, True)
        self.map_settings.setFlag(QgsMapSettings.Flag.DrawLabeling, True)
        if antialias:
            self.map_settings.setFlag(QgsMapSettings.Flag.Antialiasing, True)

        self.wgs84_to_3857 = QgsCoordinateTransform(
            QgsCoordinateReferenceSystem('EPSG:4326'),
            QgsCoordinateReferenceSystem('EPSG:3857'),
            QgsProject.instance()
        )

        root = QgsProject.instance().layerTreeRoot()
        layers = [
            node.layer() for node in root.findLayers()
            if node.isVisible() and node.layer() and node.layer().isSpatial()
        ]
        self.map_settings.setLayers(layers)

        if self.skip_empty:
            self._build_feature_index(layers, self.extent_3857)

        self.tiles_done = 0
        self.tiles_total = 0
        self.tiles_skipped = 0
        self.zoom_done = 0
        self.zoom_total = 0
        self.current_zoom = 0

        self._PROGRESS_INTERVAL = 20
        self._progress_counter = 0

    def close(self):
        self.io_pool.shutdown(wait=True)

    def _build_feature_index(self, layers, extent_3857):
        try:
            from qgis.core import QgsSpatialIndex, QgsFeature, QgsGeometry
            crs_3857 = QgsCoordinateReferenceSystem('EPSG:3857')
            idx = QgsSpatialIndex()
            fid = 0
            any_feat = False

            # Margines wokół obszaru na wypadek etykiet wystających z krawędzi
            margin = 1000.0  # 1 km w metrach
            search_extent = QgsRectangle(
                extent_3857.xMinimum() - margin, extent_3857.yMinimum() - margin,
                extent_3857.xMaximum() + margin, extent_3857.yMaximum() + margin
            )

            for lyr in layers:
                if not hasattr(lyr, 'getFeatures'):
                    continue

                # Transformacja obszaru roboczego do CRS warstwy
                lyr_crs = lyr.crs()
                if lyr_crs.authid() != 'EPSG:3857':
                    ct_to_lyr = QgsCoordinateTransform(crs_3857, lyr_crs, QgsProject.instance())
                    filter_rect = ct_to_lyr.transformBoundingBox(search_extent)
                    ct_to_3857 = QgsCoordinateTransform(lyr_crs, crs_3857, QgsProject.instance())
                else:
                    filter_rect = search_extent
                    ct_to_3857 = None

                req = QgsFeatureRequest().setFilterRect(filter_rect).setNoAttributes()

                feats_to_add = []
                for f in lyr.getFeatures(req):
                    g = f.geometry()
                    if g is None or g.isEmpty():
                        continue
                    bb = g.boundingBox()
                    if ct_to_3857 is not None:
                        bb = ct_to_3857.transformBoundingBox(bb)
                    nf = QgsFeature(fid)
                    nf.setGeometry(QgsGeometry.fromRect(bb))
                    feats_to_add.append(nf)
                    fid += 1

                if feats_to_add:
                    idx.addFeatures(feats_to_add)
                    any_feat = True

            self._feat_index = idx if any_feat else None
            if self.feedback and any_feat:
                self.feedback.pushInfo(f'Indeks przestrzenny: zaindeksowano {fid} obiektów w zadanym zasięgu.')
        except Exception as e:
            self._feat_index = None
            if self.feedback:
                self.feedback.pushInfo(f'Błąd indeksu przestrzennego: {e}')

    def render_zoom(self, zoom, tiles):
        if not tiles:
            return

        self.current_zoom = zoom
        self.zoom_total = len(tiles)
        self.zoom_done = 0

        xs = [t[0] for t in tiles]
        ys = [t[1] for t in tiles]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        step = self.metatile_size
        col = x_min
        while col <= x_max:
            row = y_min
            while row <= y_max:
                if self.feedback and self.feedback.isCanceled():
                    return
                self._render_metatile(
                    zoom,
                    row_min=row, row_max=min(row + step - 1, y_max),
                    col_min=col, col_max=min(col + step - 1, x_max)
                )
                row += step
            col += step

    def _render_metatile(self, z, row_min, row_max, col_min, col_max):
        buf = self.buffer_tiles if self.use_buffer else 0
        bx_min = col_min - buf
        bx_max = col_max + buf
        by_min = row_min - buf
        by_max = row_max + buf

        cols = bx_max - bx_min + 1
        rows = by_max - by_min + 1

        n = 2.0 ** z
        corner_tl = Tile(bx_min, by_min, z).to_point_wgs84(n)
        corner_br = Tile(bx_max + 1, by_max + 1, z).to_point_wgs84(n)
        rect_wgs84 = QgsRectangle(corner_tl, corner_br)
        rect_3857 = self.wgs84_to_3857.transformBoundingBox(rect_wgs84)

        if self.skip_empty and self._feat_index is not None:
            tw = (20037508.342789244 * 2.0) / (2.0 ** z)
            qrect = QgsRectangle(
                rect_3857.xMinimum() - tw, rect_3857.yMinimum() - tw,
                rect_3857.xMaximum() + tw, rect_3857.yMaximum() + tw
            )
            if not self._feat_index.intersects(qrect):
                n_tiles = (col_max - col_min + 1) * (row_max - row_min + 1)
                self.tiles_skipped += n_tiles
                self.tiles_done += n_tiles
                self.zoom_done += n_tiles
                self._update_progress(n_tiles)
                return

        img_w = self.TILE_W * cols
        img_h = self.TILE_H * rows

        self.map_settings.setExtent(rect_3857)
        self.map_settings.setOutputSize(QSize(img_w, img_h))

        loop = QEventLoop()
        job = QgsMapRendererParallelJob(self.map_settings)
        job.finished.connect(loop.quit)
        job.start()

        timer = QTimer()
        timer.setInterval(200)
        timer.timeout.connect(
            lambda: loop.quit() if (self.feedback and self.feedback.isCanceled()) else None
        )
        timer.start()
        loop.exec()
        timer.stop()

        if self.feedback and self.feedback.isCanceled():
            job.cancel()
            job.waitForFinished()
            return

        img = job.renderedImage()

        for xi in range(buf, cols - buf):
            for yi in range(buf, rows - buf):
                tile_x = bx_min + xi
                tile_y = by_min + yi

                tile_img = img.copy(
                    xi * self.TILE_W,
                    yi * self.TILE_H,
                    self.TILE_W,
                    self.TILE_H
                )

                if self.skip_empty and self._is_fully_transparent_np(tile_img):
                    self.tiles_skipped += 1
                    self.tiles_done += 1
                    self.zoom_done += 1
                    self._update_progress(1)
                    continue

                self.io_pool.submit(self._save_tile_worker, tile_img, tile_x, tile_y, z)

                self.tiles_done += 1
                self.zoom_done += 1
                self._update_progress(1)

    @staticmethod
    def _is_fully_transparent_np(image):
        if not image.hasAlphaChannel():
            return False
        if image.format() != QImage.Format.Format_ARGB32:
            image = image.convertToFormat(QImage.Format.Format_ARGB32)

        h = image.height()
        bpl = image.bytesPerLine()
        
        # W PyQt6 należy jawnie nadać rozmiar dla sip.voidptr
        ptr = image.constBits()
        ptr.setsize(h * bpl)
        
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bpl))
        # Sprawdzamy kanał alfa (co 4. bajt: B, G, R, [A])
        return not arr[:, 3::4].any()

    def _save_tile_worker(self, image, x, y, z):
        dir_path = os.path.join(self.output_dir, str(z), str(x))
        if dir_path not in self._created_dirs:
            os.makedirs(dir_path, exist_ok=True)
            self._created_dirs.add(dir_path)

        file_path = os.path.join(dir_path, f'{y}.{self.ext}')
        image.save(file_path, self.img_format, self.quality)

    def _update_progress(self, count):
        if not self.feedback:
            return
        self._progress_counter += count
        if self._progress_counter >= self._PROGRESS_INTERVAL:
            self._progress_counter = 0
            if self.tiles_total > 0:
                self.feedback.setProgress(int(self.tiles_done / self.tiles_total * 100))
            self.feedback.setProgressText(
                f'Zoom {self.current_zoom}: {self.zoom_done} / {self.zoom_total} kafli'
                f'  |  łącznie: {self.tiles_done} / {self.tiles_total}'
            )


class GenerateXYZMetatilesAlgorithm(QgsProcessingAlgorithm):
    INPUT_EXTENT = 'EXTENT'
    ZOOM_MIN = 'ZOOM_MIN'
    ZOOM_MAX = 'ZOOM_MAX'
    OUTPUT_DIR = 'OUTPUT_DIR'
    DPI = 'DPI'
    METATILE_SIZE = 'METATILE_SIZE'
    USE_BUFFER = 'USE_BUFFER'
    BUFFER_TILES = 'BUFFER_TILES'
    ANTIALIAS = 'ANTIALIAS'
    FORMAT = 'FORMAT'
    QUALITY = 'QUALITY'
    BACKGROUND = 'BACKGROUND'
    SKIP_EMPTY = 'SKIP_EMPTY'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterExtent(self.INPUT_EXTENT, 'Zasięg (extent)'))
        self.addParameter(QgsProcessingParameterNumber(self.ZOOM_MIN, 'Minimalny zoom', defaultValue=14, minValue=0, maxValue=22))
        self.addParameter(QgsProcessingParameterNumber(self.ZOOM_MAX, 'Maksymalny zoom', defaultValue=17, minValue=0, maxValue=22))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT_DIR, 'Katalog wyjściowy'))
        self.addParameter(QgsProcessingParameterNumber(self.DPI, 'DPI renderowania', defaultValue=96, minValue=72, maxValue=300))
        self.addParameter(QgsProcessingParameterNumber(self.METATILE_SIZE, 'Rozmiar metatile (kafle)', defaultValue=16, minValue=1, maxValue=16))
        self.addParameter(QgsProcessingParameterBoolean(self.USE_BUFFER, 'Bufor metatile', defaultValue=True))
        self.addParameter(QgsProcessingParameterNumber(self.BUFFER_TILES, 'Szerokość bufora w kaflach', defaultValue=1, minValue=0, maxValue=4))
        self.addParameter(QgsProcessingParameterBoolean(self.ANTIALIAS, 'Antyaliasing', defaultValue=True))
        self.addParameter(QgsProcessingParameterEnum(
            self.FORMAT, 'Format kafli',
            options=['WebP (zalecany, alfa)', 'PNG (32-bit, alfa)', 'JPEG'],
            defaultValue=0))
        self.addParameter(QgsProcessingParameterNumber(self.QUALITY, 'Jakość (WebP/JPEG)', defaultValue=95, minValue=1, maxValue=100))
        self.addParameter(QgsProcessingParameterColor(self.BACKGROUND, 'Kolor tła', defaultValue=QColor(255, 255, 255, 0), optional=True))
        self.addParameter(QgsProcessingParameterBoolean(self.SKIP_EMPTY, 'Pomijaj w pełni przezroczyste kafle', defaultValue=True))

    def processAlgorithm(self, parameters, context, feedback):
        extent_param = self.parameterAsExtent(parameters, self.INPUT_EXTENT, context)
        extent_crs   = self.parameterAsExtentCrs(parameters, self.INPUT_EXTENT, context)
        zoom_min     = self.parameterAsInt(parameters, self.ZOOM_MIN, context)
        zoom_max     = self.parameterAsInt(parameters, self.ZOOM_MAX, context)
        output_dir   = self.parameterAsString(parameters, self.OUTPUT_DIR, context)
        dpi          = self.parameterAsInt(parameters, self.DPI, context)
        meta_size    = self.parameterAsInt(parameters, self.METATILE_SIZE, context)
        use_buffer   = self.parameterAsBool(parameters, self.USE_BUFFER, context)
        buffer_tiles = self.parameterAsInt(parameters, self.BUFFER_TILES, context)
        antialias    = self.parameterAsBool(parameters, self.ANTIALIAS, context)
        fmt_idx      = self.parameterAsEnum(parameters, self.FORMAT, context)
        quality      = self.parameterAsInt(parameters, self.QUALITY, context)
        background   = self.parameterAsColor(parameters, self.BACKGROUND, context)
        skip_empty   = self.parameterAsBool(parameters, self.SKIP_EMPTY, context)

        img_format = ['WEBP', 'PNG', 'JPEG'][fmt_idx]

        # Konwersja zasięgu do WGS84 (na potrzeby obliczeń numeracji kafli)
        if extent_crs.authid() != 'EPSG:4326':
            xform_wgs84 = QgsCoordinateTransform(
                extent_crs,
                QgsCoordinateReferenceSystem('EPSG:4326'),
                QgsProject.instance()
            )
            extent_wgs84 = xform_wgs84.transformBoundingBox(extent_param)
        else:
            extent_wgs84 = extent_param

        # Konwersja zasięgu do EPSG:3857 (na potrzeby indeksowania obiektów)
        if extent_crs.authid() != 'EPSG:3857':
            xform_3857 = QgsCoordinateTransform(
                extent_crs,
                QgsCoordinateReferenceSystem('EPSG:3857'),
                QgsProject.instance()
            )
            extent_3857 = xform_3857.transformBoundingBox(extent_param)
        else:
            extent_3857 = extent_param

        all_tiles = tiles_for_extent(extent_wgs84, zoom_min, zoom_max)
        total = sum(len(v) for v in all_tiles.values())
        feedback.pushInfo(f'Łączna liczba kafli: {total}')

        renderer = MetatileRenderer(
            output_dir=output_dir,
            extent_3857=extent_3857,
            dpi=dpi,
            background=background,
            antialias=antialias,
            img_format=img_format,
            quality=quality,
            metatile_size=meta_size,
            use_buffer=use_buffer,
            feedback=feedback,
            skip_empty=skip_empty,
            buffer_tiles=buffer_tiles
        )
        renderer.tiles_total = total

        t_start = time.monotonic()
        try:
            for z in range(zoom_min, zoom_max + 1):
                if feedback.isCanceled():
                    break
                tiles = all_tiles.get(z, [])
                feedback.pushInfo(f'Zoom {z}: start ({len(tiles)} kafli)...')
                renderer.render_zoom(z, tiles)
        finally:
            feedback.pushInfo('Oczekiwanie na dokończenie zapisu plików na dysku...')
            renderer.close()

        elapsed = time.monotonic() - t_start
        feedback.pushInfo(f'Całkowity czas: {int(elapsed // 60)}m {int(elapsed % 60)}s')

        return {self.OUTPUT_DIR: output_dir}

    def name(self):
        return 'generate_XYZ_metatiles'

    def displayName(self):
        return 'Generuj kafle XYZ z buforem metatile (Fast 1-1)'

    def group(self):
        return 'Kafle XYZ'

    def groupId(self):
        return 'xyzmetatiles'

    def createInstance(self):
        return GenerateXYZMetatilesAlgorithm()
