#!/usr/bin/env python
# licence.py - display GPL licence

# Copyright (c) 2010-2011 Algis Kabaila. All rights reserved.
# This program or module is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public Licence as published
# by the Free Software Foundation, either version 2 of the Licence, or
# version 3 of the Licence, or (at your option) any later version. It is
# provided for educational purposes and is distributed in the hope that
# it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
# the GNU General Public Licence for more details.
'''
To do:

1. Use mayavi without traits.  http://docs.enthought.com/mayavi/mayavi/
2. Get useful parts of surf.py and ideal_gui.py
3. Move dot smoothly
4. Draw nice lines smoothly
5. Erase lines
6. Strip down
'''
import sys

from PySide.QtGui import QApplication, QMainWindow, QTextEdit, QPushButton

from PySide import QtGui, QtCore

from ui_ideal_qt import Ui_MainWindow

from ui_P_control import Ui_Form as P_control
from ui_PVE_control import Ui_Form as PVE_control

import surf
class P_widget(QtGui.QWidget, P_control):
    def __init__(self, parent=None):
        '''Mandatory initialisation of a class.'''
        super(P_widget, self).__init__(parent)
        self.setupUi(self)
        QtCore.QObject.connect(
            self.verticalSlider,
            QtCore.SIGNAL("valueChanged(int)"),
            self.slider_2_spin_box)
        QtCore.QObject.connect(
            self.doubleSpinBox,
            QtCore.SIGNAL("valueChanged(double)"),
            self.spin_box_2_slider)
    def slider_2_spin_box(self, i):
        self.doubleSpinBox.setValue(float(i))
    def spin_box_2_slider(self, f):
        self.verticalSlider.setValue(int(f))
    def debug(*args):
        print('args=%s'%(args,))

class PVE_widget(QtGui.QWidget, PVE_control):
    def __init__(self, parent=None):
        '''Mandatory initialisation of a class.'''
        super(PVE_widget, self).__init__(parent)
        self.setupUi(self)
        QtCore.QObject.connect(
            self.verticalSlider_2,
            QtCore.SIGNAL("valueChanged(int)"),
            self.slider_2_spin_box)
        QtCore.QObject.connect(
            self.doubleSpinBox_2,
            QtCore.SIGNAL("valueChanged(double)"),
            self.spin_box_2_slider)
    def slider_2_spin_box(self, i):
        self.doubleSpinBox_2.setValue(float(i)/4)
    def spin_box_2_slider(self, f):
        self.verticalSlider_2.setValue(int(f*4))

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        '''Mandatory initialisation of a class.'''
        super(MainWindow, self).__init__(parent)
        #self.setupUi(self)
        self.setupUi(self, P_widget, PVE_widget)
        #self.setupUi(self, surf.MayaviQWidget, PVE_widget)
        #self.setupUi(self, QtGui.QListView)
        #self.showButton.clicked.connect(self.fileRead)
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    frame = MainWindow()
    frame.show()
    app.exec_()
    
