from __future__ import annotations
from typing import List, Optional, Callable

from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsObject,
    QGraphicsItem, QApplication,
)
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QBrush, QFont,
    QWheelEvent, QMouseEvent, QKeyEvent,
)

from .annotation import BoundingBox

HANDLE_SIZE = 9
H = HANDLE_SIZE / 2

# Handle index constants
NW, N, NE, E, SE, S, SW, W = range(8)

COLORS_AUTO = QColor(80, 190, 255)
COLORS_MANUAL = QColor(100, 255, 140)
COLORS_SELECTED = QColor(255, 210, 0)
COLOR_TEXT_BG = QColor(0, 0, 0, 170)


class BBoxItem(QGraphicsObject):
    """One bounding box annotation as a QGraphicsObject."""

    sig_selected = pyqtSignal(object)   # self
    sig_modified = pyqtSignal(object)   # self
    sig_label_requested = pyqtSignal(object)  # self

    def __init__(self, bb: BoundingBox, img_w: int, img_h: int):
        super().__init__()
        self.bb = bb
        self.img_w = img_w
        self.img_h = img_h
        self._selected = False
        self._drag_mode: Optional[int | str] = None
        self._drag_start: Optional[QPointF] = None
        self._rect_snap: Optional[QRectF] = None

        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setZValue(1)

    # ── geometry ──────────────────────────────────────────────────

    def pix_rect(self) -> QRectF:
        x1, y1, x2, y2 = self.bb.to_pixel(self.img_w, self.img_h)
        return QRectF(x1, y1, x2 - x1, y2 - y1)

    def _handle_rects(self) -> List[QRectF]:
        r = self.pix_rect()
        cx, cy = r.center().x(), r.center().y()
        pts = [
            (r.left(), r.top()), (cx, r.top()), (r.right(), r.top()),
            (r.right(), cy),
            (r.right(), r.bottom()), (cx, r.bottom()), (r.left(), r.bottom()),
            (r.left(), cy),
        ]
        return [QRectF(px - H, py - H, HANDLE_SIZE, HANDLE_SIZE) for px, py in pts]

    def boundingRect(self) -> QRectF:
        r = self.pix_rect()
        return r.adjusted(-HANDLE_SIZE, -HANDLE_SIZE - 20, HANDLE_SIZE, HANDLE_SIZE)

    def _handle_at(self, pos: QPointF) -> Optional[int]:
        if not self._selected:
            return None
        for i, hr in enumerate(self._handle_rects()):
            if hr.contains(pos):
                return i
        return None

    # ── paint ─────────────────────────────────────────────────────

    def paint(self, painter: QPainter, option, widget=None):
        r = self.pix_rect()
        color = COLORS_SELECTED if self._selected else (
            COLORS_AUTO if self.bb.is_auto else COLORS_MANUAL
        )

        pen = QPen(color, 2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(r)

        # Label tag
        if self.bb.label:
            font = QFont("Segoe UI", 9, QFont.Weight.Bold)
            painter.setFont(font)
            fm = painter.fontMetrics()
            tag = self.bb.label
            if self.bb.confidence > 0:
                tag += f" {self.bb.confidence:.2f}"
            tw = fm.horizontalAdvance(tag)
            th = fm.height()
            bg = QRectF(r.left(), r.top() - th - 4, tw + 8, th + 4)
            painter.fillRect(bg, COLOR_TEXT_BG)
            painter.setPen(QPen(color))
            painter.drawText(QPointF(r.left() + 4, r.top() - 4), tag)

        # Handles
        if self._selected:
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.setBrush(QBrush(color))
            for hr in self._handle_rects():
                painter.drawRect(hr)

    # ── selection ─────────────────────────────────────────────────

    def set_selected(self, val: bool):
        self._selected = val
        self.prepareGeometryChange()
        self.update()

    # ── mouse ─────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            h = self._handle_at(event.pos())
            self._drag_mode = h if h is not None else "move"
            self._drag_start = event.scenePos()
            self._rect_snap = self.pix_rect()
            self.sig_selected.emit(self)
        event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.sig_label_requested.emit(self)
        event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_start is None:
            return
        delta = event.scenePos() - self._drag_start
        r = QRectF(self._rect_snap)
        dx, dy = delta.x(), delta.y()

        if self._drag_mode == "move":
            r.translate(dx, dy)
        else:
            h = self._drag_mode
            if h in (NW, W, SW):  r.setLeft(r.left() + dx)
            if h in (NE, E, SE):  r.setRight(r.right() + dx)
            if h in (NW, N, NE):  r.setTop(r.top() + dy)
            if h in (SW, S, SE):  r.setBottom(r.bottom() + dy)
            r = r.normalized()

        # Clamp
        r.setLeft(max(0.0, r.left()))
        r.setTop(max(0.0, r.top()))
        r.setRight(min(float(self.img_w), r.right()))
        r.setBottom(min(float(self.img_h), r.bottom()))

        self.prepareGeometryChange()
        self.bb.x1 = r.left() / self.img_w
        self.bb.y1 = r.top() / self.img_h
        self.bb.x2 = r.right() / self.img_w
        self.bb.y2 = r.bottom() / self.img_h
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._drag_mode is not None:
            self.sig_modified.emit(self)
        self._drag_mode = None
        self._drag_start = None
        self._rect_snap = None
        event.accept()

    def hoverEnterEvent(self, event):
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def hoverLeaveEvent(self, event):
        self.unsetCursor()


class AnnotationCanvas(QGraphicsView):
    """
    Main canvas: displays image, manages BBoxItems.
    Supports zoom (wheel), pan (middle-drag), draw new BB (left-drag on empty),
    select/resize existing BB (left-drag on item/handle).
    """
    sig_bb_selected = pyqtSignal(object)   # BoundingBox | None
    sig_bb_created = pyqtSignal(object)    # BoundingBox
    sig_bb_deleted = pyqtSignal(object)    # BoundingBox
    sig_bb_modified = pyqtSignal(object)   # BoundingBox
    sig_label_requested = pyqtSignal(object)  # BBoxItem

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(28, 28, 32)))

        self._pixmap_item = None
        self._img_w = 1
        self._img_h = 1
        self._items: List[BBoxItem] = []
        self._selected: Optional[BBoxItem] = None

        # Drawing state
        self._drawing = False
        self._draw_origin: Optional[QPointF] = None
        self._draw_ghost = None   # temp QGraphicsRectItem

        # Pan state
        self._panning = False
        self._pan_origin: Optional[QPointF] = None

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ── public API ────────────────────────────────────────────────

    def load_image(self, pixmap, annotations: List[BoundingBox]):
        self._scene.clear()
        self._items.clear()
        self._selected = None
        self._draw_ghost = None

        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._pixmap_item.setZValue(0)
        self._img_w = pixmap.width()
        self._img_h = pixmap.height()

        for bb in annotations:
            self._add_item(bb)

        self.resetTransform()
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def clear(self):
        self._scene.clear()
        self._items.clear()
        self._selected = None
        self._draw_ghost = None
        self._pixmap_item = None
        self._img_w = 1
        self._img_h = 1

    def get_annotations(self) -> List[BoundingBox]:
        return [it.bb for it in self._items]

    def add_bb(self, bb: BoundingBox):
        item = self._add_item(bb)
        self._select(item)
        self.sig_bb_created.emit(bb)

    def delete_selected(self):
        if self._selected:
            bb = self._selected.bb
            self._scene.removeItem(self._selected)
            self._items.remove(self._selected)
            self._selected = None
            self.sig_bb_selected.emit(None)
            self.sig_bb_deleted.emit(bb)

    def highlight_by_id(self, bb_id: str):
        for it in self._items:
            if it.bb.bb_id == bb_id:
                self._select(it)
                break

    def update_item(self, bb_id: str):
        for it in self._items:
            if it.bb.bb_id == bb_id:
                it.prepareGeometryChange()
                it.update()
                break

    # ── private ───────────────────────────────────────────────────

    def _add_item(self, bb: BoundingBox) -> BBoxItem:
        item = BBoxItem(bb, self._img_w, self._img_h)
        item.sig_selected.connect(self._on_item_selected)
        item.sig_modified.connect(lambda it: self.sig_bb_modified.emit(it.bb))
        item.sig_label_requested.connect(self.sig_label_requested.emit)
        self._scene.addItem(item)
        self._items.append(item)
        return item

    def _select(self, item: Optional[BBoxItem]):
        if self._selected and self._selected is not item:
            self._selected.set_selected(False)
        self._selected = item
        if item:
            item.set_selected(True)
            self.sig_bb_selected.emit(item.bb)
        else:
            self.sig_bb_selected.emit(None)

    def _on_item_selected(self, item: BBoxItem):
        self._select(item)

    # ── events ────────────────────────────────────────────────────

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_origin = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())
            hit = self._scene.itemAt(scene_pos, self.transform())
            if hit is None or hit is self._pixmap_item:
                # Begin drawing
                self._drawing = True
                self._draw_origin = scene_pos
                self._select(None)
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._panning and self._pan_origin is not None:
            delta = event.position().toPoint() - self._pan_origin
            self._pan_origin = event.position().toPoint()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
            return

        if self._drawing and self._draw_origin is not None:
            sp = self.mapToScene(event.position().toPoint())
            r = QRectF(self._draw_origin, sp).normalized()
            if self._draw_ghost:
                self._scene.removeItem(self._draw_ghost)
            pen = QPen(QColor(255, 220, 0), 2, Qt.PenStyle.DashLine)
            self._draw_ghost = self._scene.addRect(r, pen)
            self._draw_ghost.setZValue(10)
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self._pan_origin = None
            self.unsetCursor()
            return

        if self._drawing and self._draw_origin is not None:
            sp = self.mapToScene(event.position().toPoint())
            r = QRectF(self._draw_origin, sp).normalized()
            self._drawing = False
            self._draw_origin = None
            if self._draw_ghost:
                self._scene.removeItem(self._draw_ghost)
                self._draw_ghost = None
            if r.width() > 4 and r.height() > 4:
                bb = BoundingBox(
                    x1=max(0.0, r.left() / self._img_w),
                    y1=max(0.0, r.top() / self._img_h),
                    x2=min(1.0, r.right() / self._img_w),
                    y2=min(1.0, r.bottom() / self._img_h),
                    label="", is_auto=False,
                )
                self.add_bb(bb)
            return

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected()
        else:
            super().keyPressEvent(event)
