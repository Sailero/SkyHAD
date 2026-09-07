"""Read-only Qt projections of recorded HAD state (coordinates are metres)."""
from __future__ import annotations

import math
import os
from pathlib import Path
from collections.abc import Iterable

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QGraphicsItem, QGraphicsObject, QGraphicsPathItem, QGraphicsScene, QGraphicsView


SIDE_COLORS = {"red": "#ff797f", "blue": "#69b6ff", "targets": "#f2cd77"}


def ensure_fonts():
    """Qt's Windows offscreen backend has no system font database by default."""
    if QFontDatabase.families():
        return
    font_dir = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    candidates = [font_dir / name for name in ("segoeui.ttf", "segoeuib.ttf", "msyh.ttc", "msyhbd.ttc")]
    candidates += [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")]
    for path in candidates:
        if path.is_file():
            QFontDatabase.addApplicationFont(str(path))


def entity_key(entity: dict) -> tuple[str, str]:
    return str(entity.get("side", "")), str(entity.get("id", ""))


def has_position(entity: dict) -> bool:
    value = entity.get("position")
    return isinstance(value, (list, tuple)) and len(value) >= 3 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in value[:3])


def point(entity: dict, projection: str = "xy") -> QPointF:
    xyz = entity.get("position", [0, 0, 0])
    return QPointF(float(xyz[0]), -float(xyz[1 if projection == "xy" else 2]))


def cosmetic_pen(color: QColor | str, width: float = 1.0, dashed: bool = False) -> QPen:
    pen = QPen(QColor(color), width)
    pen.setCosmetic(True)
    if dashed:
        pen.setStyle(Qt.PenStyle.DashLine)
    return pen


class EntityItem(QGraphicsObject):
    """Fixed-pixel glyph whose position alone is in world coordinates."""

    def __init__(self, entity: dict, projection: str = "xy"):
        super().__init__()
        self.entity = entity
        self.projection = projection
        self.labels = False
        self.light = False
        self.flash = False
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
        self._paint_signature = None
        self.setZValue(10)
        self.update_entity(entity)

    def boundingRect(self):
        return QRectF(-22, -24, 136 if self.labels or self.isSelected() else 46, 54)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedChange:
            self.prepareGeometryChange()
        return super().itemChange(change, value)

    def shape(self):
        path = QPainterPath()
        path.addEllipse(QRectF(-13, -13, 26, 26))
        return path

    def update_entity(self, entity: dict):
        self.entity = entity
        self.setPos(point(entity, self.projection))
        signature = (entity.get("health"), entity.get("alive"), entity.get("role"),
                     tuple(entity.get("velocity") or []), self.labels, self.light, self.flash, self.isSelected())
        if signature != self._paint_signature:
            self._paint_signature = signature
            self.setToolTip(f"{entity.get('side')} {entity.get('id')} · {entity.get('role')}\nHP {entity.get('health', '—')}")
            self.update()

    def paint(self, painter, option, widget=None):
        e = self.entity
        alive = e.get("alive") is not False and e.get("alive") != 0
        if e.get("alive") is None:
            alive = True  # Unknown historical state is not evidence of death.
        color = QColor(SIDE_COLORS.get(e.get("side"), "#b9c4d5"))
        if self.light:
            color = color.darker(140)
        if not alive:
            color = QColor("#788398")
            painter.setOpacity(0.6)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.isSelected() or self.flash:
            painter.setPen(cosmetic_pen("#ffffff" if self.flash else color, 1.8))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QRectF(-16, -16, 32, 32))
        painter.setPen(cosmetic_pen(color.lighter(135), 1.1))
        painter.setBrush(color)
        role = str(e.get("role", "Attack")).lower()
        velocity = e.get("velocity") or [0, 0, 0]
        vy = float(velocity[1 if self.projection == "xy" else 2])
        vx = float(velocity[0])
        angle = math.degrees(math.atan2(-vy, vx)) if vx or vy else -90
        painter.save()
        if e.get("side") == "targets" or role in {"target", "entity"}:
            painter.drawRect(QRectF(-7, -7, 14, 14))
            painter.setPen(cosmetic_pen("#0e1523" if not self.light else "#ffffff", 1.6))
            painter.drawLine(QPointF(-4, 0), QPointF(4, 0))
            painter.drawLine(QPointF(0, -4), QPointF(0, 4))
        elif role == "unknown":
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QRectF(-7, -7, 14, 14))
            painter.setFont(QFont("Segoe UI", 9))
            painter.drawText(QPointF(-3, 4), "?")
        else:
            painter.rotate(angle)
            if "scout" in role:
                painter.drawEllipse(QRectF(-6, -6, 12, 12))
            elif "disturb" in role:
                painter.drawPolygon(QPolygonF([QPointF(8, 0), QPointF(0, -7), QPointF(-8, 0), QPointF(0, 7)]))
            else:
                painter.drawPolygon(QPolygonF([QPointF(10, 0), QPointF(-7, -6), QPointF(-3, 0), QPointF(-7, 6)]))
            if vx or vy:
                painter.setPen(cosmetic_pen(color, 1.1))
                painter.drawLine(QPointF(10, 0), QPointF(19, 0))
        painter.restore()
        health = float(e.get("health") or 0)
        max_health = float(e.get("max_health", 0) or 0)
        if max_health > 0:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#344156" if not self.light else "#ced6e1"))
            painter.drawRoundedRect(QRectF(-9, 12, 18, 2.5), 1, 1)
            painter.setBrush(color)
            painter.drawRoundedRect(QRectF(-9, 12, 18 * max(0, min(1, health / max_health)), 2.5), 1, 1)
        if not alive:
            painter.setPen(cosmetic_pen(color, 1.5))
            painter.drawLine(QPointF(-8, -8), QPointF(8, 8))
            painter.drawLine(QPointF(-8, 8), QPointF(8, -8))
        if self.labels or self.isSelected():
            painter.setFont(QFont("Segoe UI", 8))
            painter.setPen(QColor("#28384d" if self.light else "#dce6f3"))
            painter.drawText(QPointF(15, -5), f"{e.get('side', '')[:1].upper()}{e.get('id')}")
            if self.isSelected():
                painter.drawText(QPointF(15, 8), f"HP {health:.1f}" if e.get("health") is not None else "HP ?")


class BattlefieldView(QGraphicsView):
    """Pan/zoom/select XY or XZ view, with reusable items for dense replays."""

    entity_selected = Signal(str, str)

    def __init__(self, projection="xy", parent=None):
        super().__init__(parent)
        ensure_fonts()
        self.projection = projection
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.entities: dict[tuple[str, str], EntityItem] = {}
        self.layers = {"labels": False, "trails": True, "groups": True, "assignments": False, "ranges": False}
        self.light = False
        self.frame: dict = {}
        self.history: list[dict] = []
        self.metadata: dict = {}
        self.selected_key: tuple[str, str] | None = None
        self._world_rect = QRectF(-2700, -2700, 5400, 5400) if projection == "xy" else QRectF(-2700, -1100, 5400, 1200)
        self.scene().setSceneRect(self._world_rect)
        self._overlays: list[QGraphicsPathItem] = []
        self._trail_cache: dict[tuple, dict] = {}
        self._point_cache: dict[int, dict] = {}
        self._fitted = False
        self.scene().selectionChanged.connect(self._selection_changed)

    def set_world_bounds(self, frames: Iterable[dict], metadata: dict | None = None):
        self.metadata = metadata or {}
        self._trail_cache.clear()
        self._point_cache.clear()
        points = [point(e, self.projection) for frame in frames for e in frame.get("entities", []) if has_position(e)]
        if points:
            xmin, xmax = min(p.x() for p in points), max(p.x() for p in points)
            ymin, ymax = min(p.y() for p in points), max(p.y() for p in points)
            width, height = max(xmax - xmin, 1000), max(ymax - ymin, 300)
            self._world_rect = QRectF((xmin+xmax-width)/2, (ymin+ymax-height)/2, width, height).adjusted(-160, -100, 160, 100)
        self.scene().setSceneRect(self._world_rect)
        self._fitted = False
        self.fit_world()

    def fit_world(self):
        self.fitInView(self._world_rect, Qt.AspectRatioMode.KeepAspectRatio)
        self._fitted = True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fitted:
            self.fit_world()

    def wheelEvent(self, event):
        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        scale = self.transform().m11() * factor
        if 0.005 < scale < 20:
            self.scale(factor, factor)
            self._fitted = False
        event.accept()

    def _selection_changed(self):
        selected = [i for i in self.scene().selectedItems() if isinstance(i, EntityItem)]
        if selected:
            self.selected_key = entity_key(selected[0].entity)
            self.entity_selected.emit(*self.selected_key)
            self._draw_overlays()

    def select_entity(self, side: str, identifier: str):
        self.selected_key = (str(side), str(identifier))
        self.scene().blockSignals(True)
        self.scene().clearSelection()
        if self.selected_key in self.entities:
            self.entities[self.selected_key].setSelected(True)
        self.scene().blockSignals(False)
        self._draw_overlays()

    def set_layer(self, name: str, enabled: bool):
        self.layers[name] = enabled
        for item in self.entities.values():
            item.prepareGeometryChange()
            item.labels = self.layers["labels"]
            item.update()
        self._draw_overlays()

    def set_light_theme(self, enabled: bool):
        self.light = enabled
        for item in self.entities.values():
            item.light = enabled
            item.update()
        self._draw_overlays()
        self.viewport().update()

    def set_frame(self, frame: dict, history: list[dict] | None = None):
        self.frame = frame
        self.history = history or []
        current = {entity_key(e): e for e in frame.get("entities", []) if has_position(e)}
        for key in self.entities.keys() - current.keys():
            self.scene().removeItem(self.entities.pop(key))
        flashes = set()
        for event in frame.get("events", []):
            for endpoint in ("source", "target"):
                identifier = event.get(endpoint + "_id")
                side = event.get(endpoint + "_side")
                if identifier is not None:
                    for key, entity in current.items():
                        physical_id = entity.get("entity_id", entity.get("id"))
                        if str(physical_id) == str(identifier) and (side is None or str(side).lower() == key[0]):
                            flashes.add(key)
        for key, entity in current.items():
            if key not in self.entities:
                self.entities[key] = EntityItem(entity, self.projection)
                self.scene().addItem(self.entities[key])
            item = self.entities[key]
            if item.labels != self.layers["labels"]:
                item.prepareGeometryChange()
            item.labels = self.layers["labels"]
            item.light = self.light
            item.flash = key in flashes
            item.update_entity(entity)
        if self.selected_key in self.entities:
            self.entities[self.selected_key].setSelected(True)
        self._draw_overlays()

    def _path(self, path, color, width=1, dashed=False, fill=None):
        item = QGraphicsPathItem(path)
        item.setPen(cosmetic_pen(color, width, dashed))
        if fill:
            item.setBrush(fill)
        item.setZValue(-1)
        self.scene().addItem(item)
        self._overlays.append(item)

    def _draw_overlays(self):
        for item in self._overlays:
            self.scene().removeItem(item)
        self._overlays.clear()
        if self.layers["trails"]:
            history = self.history[-12:]  # A short history keeps dense battles legible.
            cache_key = tuple(id(frame) for frame in history)
            paths = self._trail_cache.get(cache_key)
            if paths is None:
                paths = {}
                for frame in history:
                    frame_id = id(frame)
                    cached = self._point_cache.get(frame_id)
                    positions = cached[1] if cached is not None else None
                    if positions is None:
                        positions = {entity_key(entity): point(entity, self.projection)
                                     for entity in frame.get("entities", []) if has_position(entity)}
                        self._point_cache[frame_id] = (frame, positions)  # Keep identity alive until eviction.
                    for key, position in positions.items():
                        if key not in paths:
                            paths[key] = QPainterPath(position)
                        else:
                            paths[key].lineTo(position)
                if len(self._trail_cache) > 256:
                    self._trail_cache.clear()
                if len(self._point_cache) > 512:
                    self._point_cache.clear()
                self._trail_cache[cache_key] = paths
            for key, path in paths.items():
                color = QColor(SIDE_COLORS.get(key[0], "#8995a8"))
                if self.light:
                    color = color.darker(140)
                color.setAlpha(140 if self.light else 90)
                self._path(path, color, 1.2)
        groups = self.frame.get("groups", {}) or {}
        for side, side_groups in groups.items():
            if not isinstance(side_groups, list):
                continue
            for group in side_groups:
                members = [self.entities[(str(side), str(i))] for i in group.get("members", []) if (str(side), str(i)) in self.entities]
                members = [item for item in members if item.entity.get("alive") is not False]
                if not members:
                    continue
                center = QPointF(sum(i.pos().x() for i in members) / len(members), sum(i.pos().y() for i in members) / len(members))
                if self.layers["groups"]:
                    path = QPainterPath()
                    radius = max([math.hypot(i.pos().x()-center.x(), i.pos().y()-center.y()) for i in members] + [40]) + 35
                    path.addEllipse(center, radius, radius)
                    color = QColor(SIDE_COLORS.get(side, "#8995a8"))
                    color.setAlpha(100)
                    self._path(path, color, 1, True)
                if self.layers["assignments"]:
                    target = self.entities.get(("targets", str(group.get("target_id"))))
                    if target:
                        path = QPainterPath(center)
                        path.lineTo(target.pos())
                        color = QColor(SIDE_COLORS.get(side, "#8995a8"))
                        color.setAlpha(150)
                        self._path(path, color, 1, True)
        if self.layers["ranges"] and self.selected_key in self.entities:
            item = self.entities[self.selected_key]
            radius = item.entity.get("attack_range", item.entity.get("range"))
            if isinstance(radius, (int, float)) and radius > 0:
                path = QPainterPath()
                path.addEllipse(item.pos(), radius, radius)
                self._path(path, "#a7b4c9", 1, True)

    def drawBackground(self, painter: QPainter, rect: QRectF):
        painter.fillRect(rect, QColor("#f8fafc" if self.light else "#0d1522"))
        # Tick spacing follows zoom; all text is drawn later in viewport coordinates.
        desired = 85 / max(self.transform().m11(), 1e-6)
        base = 10 ** math.floor(math.log10(max(desired, 1e-6)))
        spacing = next(v * base for v in (1, 2, 5, 10) if v * base >= desired)
        painter.setPen(cosmetic_pen("#e0e6ee" if self.light else "#1e2b3d", 1))
        x = math.floor(rect.left() / spacing) * spacing
        while x <= rect.right():
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            x += spacing
        y = math.floor(rect.top() / spacing) * spacing
        while y <= rect.bottom():
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            y += spacing
        painter.setPen(cosmetic_pen("#aebac9" if self.light else "#35455e", 1.3))
        painter.drawLine(QPointF(0, rect.top()), QPointF(0, rect.bottom()))
        painter.drawLine(QPointF(rect.left(), 0), QPointF(rect.right(), 0))

    def drawForeground(self, painter: QPainter, rect: QRectF):
        painter.save()
        painter.setWorldTransform(self.viewportTransform().inverted()[0], True)
        width, height = self.viewport().width(), self.viewport().height()
        painter.setPen(QColor("#33465e" if self.light else "#a9b8cc"))
        painter.setFont(QFont("Segoe UI", 9))
        title = "XY · 平面 / m" if self.projection == "xy" else "XZ · 高度剖面 / m"
        painter.drawText(QPointF(18, 25), title)
        painter.drawText(QPointF(18, height-15), "滚轮缩放 · 拖动平移 · 点击实体")
        scale = max(self.transform().m11(), 1e-6)
        visible = self.mapToScene(self.viewport().rect()).boundingRect()
        desired = 110 / scale
        base_tick = 10 ** math.floor(math.log10(max(desired, 1e-6)))
        spacing = next(v * base_tick for v in (1, 2, 5, 10) if v * base_tick >= desired)
        tick = math.ceil(visible.left() / spacing) * spacing
        while tick <= visible.right():
            x_tick = self.mapFromScene(QPointF(tick, 0)).x()
            if 55 < x_tick < width-50:
                painter.drawText(QPointF(x_tick-14, height-39), f"{tick:g}")
            tick += spacing
        y_desired = (38 if self.projection == "xz" else 110) / scale
        y_base = 10 ** math.floor(math.log10(max(y_desired, 1e-6)))
        y_spacing = next(v * y_base for v in (1, 2, 5, 10) if v * y_base >= y_desired)
        tick = math.ceil(visible.top() / y_spacing) * y_spacing
        while tick <= visible.bottom():
            y_tick = self.mapFromScene(QPointF(0, tick)).y()
            if 58 < y_tick < height-55:
                painter.drawText(QPointF(8, y_tick-4), f"{-tick:g}")
            tick += y_spacing
        metres = 100 / scale
        base = 10 ** math.floor(math.log10(metres))
        length = max(v * base for v in (1, 2, 5) if v * base <= metres)
        pixels = length * scale
        painter.setPen(cosmetic_pen("#4b607c" if self.light else "#a9b8cc", 1.5))
        x, y = width - pixels - 22, height - 22
        painter.drawLine(QPointF(x, y), QPointF(x+pixels, y))
        painter.drawLine(QPointF(x, y-4), QPointF(x, y+4))
        painter.drawLine(QPointF(x+pixels, y-4), QPointF(x+pixels, y+4))
        painter.drawText(QPointF(x, y-8), f"{length:g} m")
        if self.layers["ranges"]:
            painter.drawText(QPointF(18, 44), "作用范围为投影；命中以记录事件为准")
        painter.restore()
