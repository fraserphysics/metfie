#!/usr/bin/env python
# Derived from licence.py by Algis Kabaila.

# Copyright (c) 2010-2011 Algis Kabaila.  Copyright 2013 Andrew Fraser
# and Los Alamos National Laboratory. All rights reserved.  This
# program or module is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public Licence as
# published by the Free Software Foundation, either version 2 of the
# Licence, or version 3 of the Licence, or (at your option) any later
# version. It is provided for educational purposes and is distributed
# in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public Licence for more
# details.
'''
Reference:  http://docs.enthought.com/mayavi/mayavi/

'''
from PySide.QtGui import QApplication, QMainWindow, QWidget
from ui_PVE_control import Ui_Form as PVE_control
class variable:
    '''A class that collects, for P, v, E or S the spin box, slider
    and button widgets and their service routines.
    '''
    def __init__(
            self,    # variable instance
            spin,    # Qt spin box widget
            slide,   # Qt slider widget
            button,  # Qt radio button widget
            name,    # One character string \in 'PvES'
            factor,  # (variable value)/(spin box value)
            state):  # Connection to other variables and GUI
        assert name in 'PvES'
        self.spin = spin
        self.slide = slide    # Goes 0 to 99
        self.button = button
        self.name = name
        self.factor = factor
        self.state = state    # Not fully developed yet
        self.min = float(self.spin.minimum())
        self.max = float(self.spin.maximum())
        self.value = spin.value()*factor
    def spin_move(self,  # variable instance
                    f):  # value from spin box
        '''Interrupt service routine for spin box value change.  Sends new
        value to slider and calls state.update().
        '''
        self.value = f*self.factor
        self.frac = (f - self.min)/(self.max - self.min)
        i = max(0, min(99, int(self.frac*99)))
        self.slide.blockSignals(True) # So setValue won't trigger slide_move
        self.slide.setValue(i)
        self.slide.blockSignals(False)
        self.state.update(self.name, value=self.value)
    def slide_move(self, # variable instance
                    i):  # value from slider
        '''Interrupt service routine for slider value change.  Sends new
        value to spin box and calls state.update().
        '''
        self.frac = float(i)/float(99)
        f = self.min + self.frac*(self.max - self.min)
        self.spin.blockSignals(True) # So setValue won't trigger spin_move
        self.spin.setValue(f)
        self.spin.blockSignals(False)
        self.value = f*self.factor
        self.state.update(self.name, value=self.value)
    def set_value(self,       # variable instance
                  v,          # New value
                  force=False # Do everything even if value already right
                  ):
        '''Called by state.update().  Sets self.value, slider and spin,
        and returns quantized number from spin.
        '''
        if self.value != v or force: # Skip if new value same as old
            self.value = v
            f = v/self.factor
            self.spin.blockSignals(True)  #  Prevent triggering spin_move
            self.spin.setValue(f)
            self.spin.blockSignals(False)
            self.frac = (f - self.min)/(self.max - self.min)
            i = max(0, min(99, int(self.frac*99)))
            self.slide.blockSignals(True) # Prevent triggering slide_move
            self.slide.setValue(i)
            self.slide.blockSignals(False)
        return self.spin.value()*self.factor # Return a quantized value
    def bounds_check(self, v):
        t = v/self.factor
        if t > self.max or t < self.min:
            return False
        return True
class state:
    '''Contains links to GUI and present and past values of state variables.
    Has methods for moving on the EOS sub-manifold.  Keeps 3 sets of
    values:

    self.displayed_values: Correspond to quantized values in spin blocks.
    
    self.values: A set of values used for calculation in response to user
        manipulation of the gui.
    
    self.old_values: These values are on the EOS to within the precision
        of floating point calculation and they are within the bounds given
        by the spin boxes which are also the bounds on the surface plot.

    '''
    def initial_values(self,  # state instance
                       ):
        '''Set  3 value dictionaries using P and v values from self.var_dict.
        This is separate from __init__ because var_dict isn't ready when
        __init__ is called.
        '''
        import mayavi.mlab as ML
        self.values = {}
        self.displayed_values = {}
        for s, var in self.var_dict.items():
            self.values[s] = var.value
        self.vP(None, None)  # Set values['E'] and values['S'] to be
                             #consistent with v and P
        for s, var in self.var_dict.items():
            v = self.values[s]
            self.displayed_values[s] = var.set_value(v, force=True)
        self.old_values = self.values.copy()
        self.new_constant()
        # Get coordinates for the displayed state point + size
        args = tuple(self.var_dict[s].frac for s in 'PvE') + (.05,)
        # Initialize the displayed state point
        self.vis.point =  ML.points3d(
            *args, scale_factor=1.0, figure=self.vis.scene.mayavi_scene)
    def __init__(self,     # state instance
                 var_dict, # Dictionary that will hold variable instances
                 vis):     # mayavi visualization instance
        import eos
        self.EOS = eos.ideal()       # Methods for EOS constraints
        self.var_dict = var_dict
        self.vis = vis
        self.vis.curve = None
        self.curve_data = []
        self.dispatch = {                # Map GUI actions to methods
            #(Moved, constant): Method,
            ('v',    'P'):      self.vP,
            ('E',    'P'):      self.EP,
            ('P',    'v'):      self.vP,
            ('E',    'v'):      self.Ev,
            ('P',    'E'):      self.EP,
            ('v',    'E'):      self.Ev,
            ('P',    'S'):      self.PS,
            ('v',    'S'):      self.vS,
            ('E',    'S'):      self.ES
            }
    def PS(self, s, button):
        self.values['v'],self.values['E'] = self.EOS.SP2vE(
            self.values['S'],self.values['P'])
    def vS(self, s, button):
        self.values['P'], self.values['E'] = self.EOS.Sv2PE(
            self.values['S'],self.values['v'])
    def ES(self, s, button):
        self.values['P'],self.values['v'] = self.EOS.SE2Pv(
            self.values['S'], self.values['E'])
    def Ev(self, s, button):
        E = self.values['E']
        v = self.values['v']
        self.values['P'] = self.EOS.Ev2P(E, v)
        self.values['S'] = self.EOS.Ev2S(E,v)
    def EP(self, s, button):
        E = self.values['E']
        P = self.values['P']
        v = self.EOS.PE2v(P, E)
        self.values['v'] = v
        self.values['S'] = self.EOS.Ev2S(E,v)
    def vP(self, s, button):
        v = self.values['v']
        P = self.values['P']
        E = self.EOS.Pv2E(P, v)
        self.values['E'] = E
        self.values['S'] = self.EOS.Ev2S(E,v)
    def shaw_eos(self,     # state instance
                ):
        import eos
        self.EOS = eos.shaw()
    def ideal_eos(self,     # state instance
                ):
        import eos
        self.EOS = eos.ideal()
    def new_constant(self # state instance
                     ):
        '''Find which radio button is checked and then do update
        '''
        for s in 'PvES':
            if self.var_dict[s].button.isChecked():
                self.constant = s
                self.update(s, button=True)
                if self.vis.curve != None:
                    self.vis.curve.remove()
                    self.curve_data = []
                self.vis.curve = None
                if s == 'P':
                    self.u_key = lambda v: v[1] # Sort on v
                else:
                    self.u_key = lambda v: v[0] # Sort on P
                self.unique = set([])           # set of keys
                return
        assert False
    def update(self,            # state instance
               s,               # key for manipulated variable
               value = None,
               button = False   # Flag for button event
               ):
        '''Manipulation of the GUI initiates the following sequence:

        1. Change the value and display of the manipulated variable

        2. This method is called

        This method continues the response as follows:

        3. If a radio button was pressed, set old_values['constant'], then
           propagate old_values to displayed values and values and return

        4. Calculate new self.values consistent with selected constant and
           changed variable

        5. If new values are in bounds, copy them to self.old_values and
           propagate them to the GUI/display and return

        6. If the new values are out of bounds, revert to self.old_values
        '''
        import numpy as np
        import mayavi.mlab as ML
        def revert():
            '''Revert to self.old_values
            '''
            for t, var in self.var_dict.items():
                self.displayed_values[t] = var.set_value(self.old_values[t])
            self.values = self.old_values.copy()
            return
        if button:
            self.old_values['constant'] = s
            revert() # Don't move, use old_values
            return
        key = (s, self.constant)
        if key not in self.dispatch:
            print('Pushing %s with constant %s has no effect'%key)
            self.var_dict[s].set_value(self.old_values[s])
            return
        self.values[s] = value
        self.dispatch[key](s, button) # Calculate effect on other variables
        # Now self.values is on the EOS.  Check that it is in bounds
        for t, var in self.var_dict.items():
            if not var.bounds_check(self.values[t]):
                revert()
                return
        self.old_values = self.values.copy()
        for t, var in self.var_dict.items(): # Update display
            if t != s:
                self.displayed_values[t] = var.set_value(self.values[t])
        # Move the displayed state point
        fracs = list(self.var_dict[s].frac for s in 'PvE')
        self.vis.point.mlab_source.set(x=fracs[0], y=fracs[1], z=fracs[2])
        # Add point to curve if it is new
        key = self.u_key(fracs)
        if key in self.unique: return
        self.unique.add(key)
        self.curve_data.append(fracs)
        if len(self.curve_data) == 1: return
        if self.vis.curve != None: self.vis.curve.remove()
        self.curve_data.sort(key=self.u_key)
        x,y,z = (np.array(self.curve_data)[:,i] for i in (0,1,2))
        self.vis.curve = ML.plot3d(
            x, y, z, figure=self.vis.scene.mayavi_scene, tube_radius=0.01)
        return

class PVE_widget(QWidget, PVE_control):
    def __init__(self, parent=None, vis_widget=None):
        '''Mandatory initialisation of a class.'''
        super(PVE_widget, self).__init__(parent)
        self.setupUi(self)
        var_dict = {}
        self.state = state(var_dict, vis_widget.visualization)
        self.Ideal_button.clicked.connect(self.state.ideal_eos)
        self.Shaw_button.clicked.connect(self.state.shaw_eos)
        for spin, slide, button, name, factor in (
# spin                 slide                  button             name  factor
(self.doubleSpinBox_P, self.verticalSlider_P, self.radioButton_P, 'P', 1e10),
(self.doubleSpinBox_v, self.verticalSlider_v, self.radioButton_v, 'v', 1e-6),
(self.doubleSpinBox_E, self.verticalSlider_E, self.radioButton_E, 'E', 1e3),
(self.doubleSpinBox_S, self.verticalSlider_S, self.radioButton_S, 'S', 1.0)):
            var = variable(spin, slide, button, name, factor, self.state)
            var_dict[name] = var
            slide.valueChanged.connect(var.slide_move)
            spin.valueChanged.connect(var.spin_move)
            button.clicked.connect(self.state.new_constant)
        self.state.initial_values() # Set up state information from var_dict

import surf
from ui_eos_qt import Ui_MainWindow
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        '''Mandatory initialisation of a class.'''
        super(MainWindow, self).__init__(parent)
        self.setupUi(self, surf.MayaviQWidget, PVE_widget)

if __name__ == '__main__':
    app = QApplication.instance() # traitsui.api has created app
    frame = MainWindow()
    frame.show()
    app.exec_()
