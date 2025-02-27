### This tool is adapted from Laurenz Kremeyer, Siwick group, McGill University, Montreal

import sys
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QKeySequence
from PyQt5 import QtWidgets
import numpy as np
from re import findall

from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "filepath", help="filepath to h5 file saved from Dataset from uedhhlib"
    )
    args = parser.parse_args()
    return args


class DataPicker(QtWidgets.QMainWindow):
    def __init__(self, timestamps, intensities, loaded_files, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.timestamps = timestamps
        self.loaded_files = loaded_files
        self.lris = []
        self.shortcut_add_lri = QtWidgets.QShortcut(QKeySequence("+"), self)
        self.shortcut_add_lri.activated.connect(self.add_lri)
        self.shortcut_remove_lri = QtWidgets.QShortcut(QKeySequence("-"), self)
        self.shortcut_remove_lri.activated.connect(self.remove_lri)
        self.shortcut_print_rule = QtWidgets.QShortcut(QKeySequence("space"), self)
        self.shortcut_print_rule.activated.connect(self.print_rule)

        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.plot = self.graphics_layout.addPlot(
            y=intensities, axisItem={"bottom": pg.DateAxisItem()}
        )
        self.plot.setLimits(
            xMin=0,
            xMax=len(self) - 1,
            yMin=self.plot.viewRange()[1][0],
            yMax=self.plot.viewRange()[1][1],
        )
        self.setCentralWidget(self.graphics_layout)
        self.show()

    def __len__(self):
        return len(self.timestamps)

    @pyqtSlot()
    def add_lri(self):
        width = int(len(self) / 10)
        if not self.lris:
            mi, ma = 0, width
        else:
            mi = self.lri_regions[-1][-1] + int(width / 2)
            ma = mi + width
        lri = pg.LinearRegionItem([mi, ma])
        self.lris.append(lri)
        self.plot.addItem(lri)

    @pyqtSlot()
    def remove_lri(self):
        pass

    @pyqtSlot()
    def print_rule(self):
        output = "ignored_files = ["
        for region in self.lri_regions:
            if region[0] < 0:
                region[0] = 0
            if region[0] > len(self.timestamps):
                region[0] = len(self.timestamps - 1)
            if region[1] < 0:
                region[1] = 0
            if region[1] > len(self.timestamps):
                region[1] = len(self.timestamps - 1)

            def _ignored_image_from_filename(filename):

                extracted_identifiers = findall(
                    r"Cycle\s*(\d+).*?_(\d+,\d+).*?_Frm(\d+)", str(filename)
                )[0]

                return (
                    int(extracted_identifiers[0]),
                    extracted_identifiers[1].replace(",", "."),
                    int(extracted_identifiers[2]),
                )

            for file_name in self.loaded_files[region[0] : region[1] + 1]:
                _cycle, stage_pos, frame_no = _ignored_image_from_filename(file_name)
                output += f"({_cycle}, {stage_pos}, {frame_no}), "

        # remove last space and comma separator and close list
        output = output.rstrip(", ") + "]"
        print(output)

    @property
    def lri_regions(self):
        return [
            (int(np.rint(lri.getRegion()[0])), int(np.rint(lri.getRegion()[1])))
            for lri in self.lris
        ]


if __name__ == "__main__":
    import sys
    import h5py

    args = parse_args()
    with h5py.File(args.filepath, "r") as hf:
        timestamps = hf["real_time/timestamps"][()]
        intensities = hf["real_time/intensity"][()]
        loaded_files = hf["real_time/loaded_files"][()]

    app = QtWidgets.QApplication(sys.argv)
    win = DataPicker(timestamps, intensities, loaded_files)
    sys.exit(app.exec_())
