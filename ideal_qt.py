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
Reference:  http://docs.enthought.com/mayavi/mayavi/
To do:

0. Constrain sliders to move on EOS
1. Move dot smoothly
2. Draw nice lines smoothly
3. Erase lines
4. Strip down
'''
import sys

from PySide.QtGui import QApplication, QMainWindow, QTextEdit, QPushButton

from PySide import QtGui, QtCore

from ui_PVE_control import Ui_Form as PVE_control

import surf
class variable:
    def __init__(self, spin, slide, button, name, factor):
        assert name in 'PvES'
        self.spin = spin          # Holds value/factor
        self.slide = slide        # Goes 0 to 99
        self.button = button
        self.factor = factor      # Multiplier for spin
        self.name = name
        self.min = float(self.spin.minimum())
        self.max = float(self.spin.maximum())
    def spin_value(self, f):
        frac = (f - self.min)/(self.max - self.min)
        i = max(0, min(99, int(frac*99)))
        self.slide.setValue(i)
    def slide_value(self, i):
        frac = float(i)/float(99)
        f = self.min + frac*(self.max - self.min)
        self.spin.setValue(f)
class state:
    def __init__(self, var_dict):
        self.var_dict = var_dict
    def new_constant(self):
        for s in 'PvES':
            if self.var_dict[s].button.isChecked():
                self.constant = s
                return
        assert False
        
class PVE_widget(QtGui.QWidget, PVE_control):
    def __init__(self, parent=None):
        '''Mandatory initialisation of a class.'''
        super(PVE_widget, self).__init__(parent)
        self.setupUi(self)
        var_dict = {}
        self.state = state(var_dict)
        for spin, slide, button, name, factor in (
(self.doubleSpinBox_P, self.verticalSlider_P, self.radioButton_P, 'P', 1e10),
(self.doubleSpinBox_v, self.verticalSlider_v, self.radioButton_v, 'v', 1e-6),
(self.doubleSpinBox_E, self.verticalSlider_E, self.radioButton_E, 'E', 1e3),
(self.doubleSpinBox_S, self.verticalSlider_S, self.radioButton_S, 'S', 1.0)):
            var = variable(spin, slide, button, name, factor)
            for key in spin, slide, button, name:
                var_dict[key] = var
            slide.valueChanged.connect(var.slide_value)
            spin.valueChanged.connect(var.spin_value)
            button.clicked.connect(self.state.new_constant)
        self.state.new_constant()

from ui_ideal_qt import Ui_MainWindow
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        '''Mandatory initialisation of a class.'''
        super(MainWindow, self).__init__(parent)
        #self.setupUi(self, P_widget, PVE_widget)
        self.setupUi(self, surf.MayaviQWidget, PVE_widget)

def calc_PVE():
    import ideal_eos
    import numpy as np
    
    EOS = ideal_eos.EOS()
    P, v = np.mgrid[1e10:4e10:20j, 1e-6:4e-6:20j]
    P = P.T
    v = v.T
    E = EOS.Pv2E(P,v)
    ranges = []
    for a in (P,v,E):
        ranges += [a.min(),a.max()]
    scale = lambda z: (z-z.min())/(z.max()-z.min())
    return (ranges, scale(P), scale(v), scale(E))
if __name__ == '__main__':
    #app = QApplication(sys.argv)
    app = QtGui.QApplication.instance() # traitsui.api has created app
    frame = MainWindow()
    frame.show()
    app.exec_()
