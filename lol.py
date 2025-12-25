import sys
import random
from dataclasses import dataclass
from typing import List, Tuple, Any, Optional
from rtree import index
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None


@dataclass
class Rect:
    minx: float
    miny: float
    maxx: float
    maxy: float


class RTree:
    def __init__(self):
        p = index.Property()
        p.dimension = 2
        p.storage = index.RT_Memory
        self.idx = index.Index(properties=p)
        self.objects = {}
        self._id_counter = 0
        self.all_mbrs = []

    def insert(self, x, y, obj):
        bbox = (x, y, x, y)
        self.objects[self._id_counter] = obj
        self.idx.insert(self._id_counter, bbox)
        if self._id_counter % 50 == 0:
            self.all_mbrs.append([x, y, x, y])
        elif self.all_mbrs:
            m = self.all_mbrs[-1]
            m[0], m[1] = min(m[0], x), min(m[1], y)
            m[2], m[3] = max(m[2], x), max(m[3], y)
        self._id_counter += 1

    def range_search(self, rect):
        bbox = (rect.minx, rect.miny, rect.maxx, rect.maxy)
        hits = self.idx.intersection(bbox)
        return [self.objects[i] for i in hits]



TYPES = ['restaurant', 'house', 'school', 'mall', 'bank', 'shop']
RESTAURANT_FIRST = ['Чайка', 'Сливки', 'Зёрна', 'Олива', 'Сосна', 'Лавка', 'Вилочка', 'Ложка', 'Капучино', 'Вкус']
RESTAURANT_SECOND = ['Бистро', 'Кафе', 'Гриль', 'Ланч', 'Диван', 'Терраса', 'Пекарня', 'Ресторан', 'Столовая']


def generate_points(n: int, size_km=100.0):
    np.random.seed(42)
    xs = np.random.rand(n) * size_km
    ys = np.random.rand(n) * size_km
    types = np.random.choice(TYPES, size=n)
    points = []
    for i in range(n):
        obj = {'id': i, 'type': types[i], 'x': float(xs[i]), 'y': float(ys[i])}
        if types[i] == 'restaurant':
            obj['rating'] = random.randint(1, 5)
            obj['name'] = f"{random.choice(RESTAURANT_FIRST)} {random.choice(RESTAURANT_SECOND)}"
        points.append(obj)
    return points

class ResultsWindow(QtWidgets.QDialog):
    def __init__(self, data, center_pt, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
        self.windowFlags() | QtCore.Qt.WindowMinimizeButtonHint | QtCore.Qt.WindowMaximizeButtonHint)
        self.setWindowTitle("Ближайшие рестораны")
        self.resize(800, 500)
        self.data = data
        self.center_pt = center_pt
        self.parent_ref = parent

        for r in self.data:
            r['dist'] = np.sqrt((r['x'] - center_pt[0]) ** 2 + (r['y'] - center_pt[1]) ** 2)


        self.data.sort(key=lambda x: x['dist'])

        layout = QtWidgets.QVBoxLayout(self)
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["ID", "Название", "Рейтинг", "Координаты", "Расстояние (км)"])
        self.table.setSortingEnabled(True)
        self.table.itemDoubleClicked.connect(self.on_row_click)
        layout.addWidget(self.table)
        self.fill_table()

    def fill_table(self):
        self.table.setRowCount(len(self.data))
        for i, r in enumerate(self.data):
            # Создаем объекты ячеек
            item_id = QtWidgets.QTableWidgetItem(str(r['id']))
            item_name = QtWidgets.QTableWidgetItem(r.get('name', ''))
            item_rating = QtWidgets.QTableWidgetItem(str(r.get('rating', 0)))
            item_coords = QtWidgets.QTableWidgetItem(f"{r['x']:.2f}, {r['y']:.2f}")
            item_dist = QtWidgets.QTableWidgetItem(f"{r['dist']:.2f}")

            # Устанавливаем флаги "Только для чтения" для каждой ячейки
            # Мы оставляем ItemIsEnabled и ItemIsSelectable, но не добавляем ItemIsEditable
            for item in [item_id, item_name, item_rating, item_coords, item_dist]:
                item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)

            # Вставляем ячейки в таблицу
            self.table.setItem(i, 0, item_id)
            self.table.setItem(i, 1, item_name)
            self.table.setItem(i, 2, item_rating)
            self.table.setItem(i, 3, item_coords)
            self.table.setItem(i, 4, item_dist)

    def on_row_click(self, item):
        row = item.row()
        rest_id = int(self.table.item(row, 0).text())
        for r in self.data:
            if r['id'] == rest_id:
                self.parent_ref.highlight_single_restaurant(r)
                break


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Geo-Analytic System (R-Tree)')
        self.resize(1400, 900)

        self.tree = RTree()
        self.points = []
        self.restaurants = []
        self.current_radius = 5.0
        self.search_center = (50.0, 50.0)

        self.search_history = []
        self.route_points = []
        self.mbr_items = []
        self.history_items = []
        self.single_highlight = []
        self.heatmap_item = None
        self.route_line = None

        self.setup_ui()
        self.apply_styles()

    def setup_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        map_layout = QtWidgets.QVBoxLayout()
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.setBackground('#1e1e1e')

        self.plot_widget.getViewBox().setLimits(xMin=0, xMax=100, yMin=0, yMax=100)
        self.plot_widget.setRange(xRange=[0, 100], yRange=[0, 100], padding=0)

        self.scatter = pg.ScatterPlotItem(size=3)
        self.plot_widget.addItem(self.scatter)
        self.plot_widget.scene().sigMouseClicked.connect(self.on_map_click)
        map_layout.addWidget(self.plot_widget)
        main_layout.addLayout(map_layout, 3)

        side = QtWidgets.QVBoxLayout()

        side.addWidget(QtWidgets.QLabel("<b>Инструменты:</b>"))
        zoom_box = QtWidgets.QHBoxLayout()
        btn_in = QtWidgets.QPushButton("Zoom +")
        btn_out = QtWidgets.QPushButton("Zoom -")
        btn_in.clicked.connect(self.zoom_in)
        btn_out.clicked.connect(self.zoom_out)
        zoom_box.addWidget(btn_in)
        zoom_box.addWidget(btn_out)
        side.addLayout(zoom_box)

        side.addWidget(QtWidgets.QLabel("<b>Слои:</b>"))
        self.chk_mbr = QtWidgets.QCheckBox("Показывать MBR дерева")
        self.chk_mbr.stateChanged.connect(self.update_mbr_visibility)
        side.addWidget(self.chk_mbr)

        side.addWidget(QtWidgets.QPushButton("Тепловая карта", clicked=self.show_heatmap))
        side.addWidget(QtWidgets.QPushButton("Скрыть карту", clicked=self.hide_heatmap))

        side.addWidget(QtWidgets.QLabel("<b>Поиск по ID:</b>"))
        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText("Введите ID заведения...")
        side.addWidget(self.search_edit)
        btn_find_id = QtWidgets.QPushButton("Найти на карте")
        btn_find_id.clicked.connect(self.find_by_id)
        side.addWidget(btn_find_id)

        side.addWidget(QtWidgets.QLabel("<b>История:</b>"))
        self.hist_list = QtWidgets.QListWidget()
        self.hist_list.itemClicked.connect(self.load_history)
        side.addWidget(self.hist_list)

        self.route_lbl = QtWidgets.QLabel("Маршрут: 0.00 км (Shift+Клик)")
        side.addWidget(self.route_lbl)
        side.addWidget(QtWidgets.QPushButton("Сбросить путь", clicked=self.clear_route))

        side.addWidget(QtWidgets.QLabel("<b>Генерация:</b>"))
        self.spin_n = QtWidgets.QSpinBox()
        self.spin_n.setRange(1000, 2000000)
        self.spin_n.setValue(20000)
        side.addWidget(self.spin_n)
        side.addWidget(QtWidgets.QPushButton("Сгенерировать карту", clicked=self.on_generate))

        side.addStretch()
        main_layout.addLayout(side, 1)

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; color: white; }
            QPushButton { background-color: #3c3f41; padding: 6px; border: 1px solid #555; }
            QListWidget, QLineEdit, QSpinBox { background-color: #1e1e1e; color: #ccc; }
        """)

    def zoom_in(self):
        self.plot_widget.getViewBox().scaleBy(s=(0.8, 0.8))

    def zoom_out(self):
        self.plot_widget.getViewBox().scaleBy(s=(1.25, 1.25))

    def on_generate(self):
        n = self.spin_n.value()
        self.points = generate_points(n)
        self.restaurants = [p for p in self.points if p['type'] == 'restaurant']
        self.tree = RTree()
        for p in self.points: self.tree.insert(p['x'], p['y'], p)
        self.scatter.setData(x=[p['x'] for p in self.points], y=[p['y'] for p in self.points],
                             brush=pg.mkBrush(80, 80, 80, 100))
        if self.chk_mbr.isChecked(): self.update_mbr_visibility(QtCore.Qt.Checked)

    def on_map_click(self, event):
        pos = self.plot_widget.getViewBox().mapSceneToView(event.scenePos())
        if event.modifiers() & QtCore.Qt.ShiftModifier:
            self.add_route_point(pos.x(), pos.y())
        else:
            # Выбор радиуса при клике на карту
            val, ok = QtWidgets.QInputDialog.getDouble(
                self, "Радиус поиска", "Введите радиус (км):",
                value=self.current_radius, min=0.1, max=10.0, decimals=1
            )
            if ok:
                self.current_radius = val
                self.search_center = (pos.x(), pos.y())
                self.run_search()

    def run_search(self):
        if not self.points: return
        r = self.current_radius
        x, y = self.search_center
        rect = Rect(x - r, y - r, x + r, y + r)
        results = self.tree.range_search(rect)
        found_rests = [p for p in results if
                       p['type'] == 'restaurant' and ((p['x'] - x) ** 2 + (p['y'] - y) ** 2) <= r ** 2]

        self.visualize_search_results(x, y, r, found_rests)
        msg = f"({x:.1f}, {y:.1f}) R={r}км | {len(found_rests)} шт."
        self.hist_list.insertItem(0, msg)
        self.search_history.insert(0, {'x': x, 'y': y, 'r': r, 'res': found_rests})
        if found_rests:
            self.res_win = ResultsWindow(found_rests, self.search_center, self)
            self.res_win.show()

    def find_by_id(self):
        txt = self.search_edit.text().strip()
        if not txt.isdigit(): return
        idn = int(txt)

        found_item = None
        for r in self.restaurants:
            if r['id'] == idn:
                found_item = r
                break

        if found_item:
            self.highlight_single_restaurant(found_item)
        else:
            # ДОБАВЛЕНО: Окно с ошибкой
            QtWidgets.QMessageBox.warning(self, "Ошибка поиска", f"Заведение с ID {idn} не найдено на карте.")

    def highlight_single_restaurant(self, r):
        for it in self.single_highlight: self.plot_widget.removeItem(it)
        self.single_highlight.clear()
        sp = pg.ScatterPlotItem(x=[r['x']], y=[r['y']], size=15, pen=pg.mkPen('w', width=3),
                                brush=pg.mkBrush(255, 255, 0, 220))
        self.plot_widget.addItem(sp)
        self.single_highlight.append(sp)
        self.plot_widget.setXRange(max(0, r['x'] - 5), min(100, r['x'] + 5))
        self.plot_widget.setYRange(max(0, r['y'] - 5), min(100, r['y'] + 5))

    def show_heatmap(self):
        self.hide_heatmap()
        if not self.restaurants: return
        res = 100
        heat, _, _ = np.histogram2d(
            [r['x'] for r in self.restaurants], [r['y'] for r in self.restaurants],
            bins=res, range=[[0, 100], [0, 100]]
        )
        if gaussian_filter:
            heat = gaussian_filter(heat, sigma=2.5)

        self.heatmap_item = pg.ImageItem(heat)
        self.heatmap_item.setOpacity(0.55)
        self.heatmap_item.setRect(QtCore.QRectF(0, 0, 100, 100))
        colors = [(0, 0, 255), (0, 255, 255), (255, 255, 255), (255, 255, 0), (255, 0, 0)]
        cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 5), color=colors)
        self.heatmap_item.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
        self.plot_widget.addItem(self.heatmap_item)

    def hide_heatmap(self):
        if self.heatmap_item:
            self.plot_widget.removeItem(self.heatmap_item)
            self.heatmap_item = None

    def visualize_search_results(self, x, y, r, rests):
        for it in self.history_items: self.plot_widget.removeItem(it)
        self.history_items.clear()
        circle = QtWidgets.QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r)
        circle.setPen(pg.mkPen(0, 255, 0, 150, width=2))
        self.plot_widget.addItem(circle)
        self.history_items.append(circle)
        if rests:
            sp = pg.ScatterPlotItem(x=[p['x'] for p in rests], y=[p['y'] for p in rests],
                                    brush=pg.mkBrush(0, 255, 0), size=10)
            self.plot_widget.addItem(sp)
            self.history_items.append(sp)

    def update_mbr_visibility(self, state):
        for it in self.mbr_items: self.plot_widget.removeItem(it)
        self.mbr_items.clear()
        if state == QtCore.Qt.Checked:
            for m in self.tree.all_mbrs:
                rect = QtWidgets.QGraphicsRectItem(m[0], m[1], m[2] - m[0], m[3] - m[1])
                rect.setPen(pg.mkPen(0, 180, 255, 80))
                self.plot_widget.addItem(rect)
                self.mbr_items.append(rect)

    def load_history(self, item):
        data = self.search_history[self.hist_list.row(item)]
        self.search_center = (data['x'], data['y'])
        self.current_radius = data['r']
        self.visualize_search_results(data['x'], data['y'], data['r'], data['res'])

    def add_route_point(self, x, y):
        x, y = max(0, min(100, x)), max(0, min(100, y))
        self.route_points.append((x, y))
        if len(self.route_points) > 1:
            if self.route_line: self.plot_widget.removeItem(self.route_line)
            pts = np.array(self.route_points)
            self.route_line = pg.PlotDataItem(pts[:, 0], pts[:, 1], pen=pg.mkPen('r', width=3))
            self.plot_widget.addItem(self.route_line)
            dist = sum(np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1)))
            self.route_lbl.setText(f"Маршрут: {dist:.2f} км")

    def clear_route(self):
        self.route_points = []
        if self.route_line: self.plot_widget.removeItem(self.route_line)
        self.route_lbl.setText("Маршрут: 0.00 км (Shift+Клик)")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())