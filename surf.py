''' surf.py derived from Enthought example qt_embedding.py
'''
# First, and before importing any Enthought packages, set the ETS_TOOLKIT
# environment variable to qt4, to tell Traits that we will use Qt.
import os
os.environ['ETS_TOOLKIT'] = 'qt4'

# To be able to use PySide or PyQt4 and not run in conflicts with traits,
# we need to import QtGui and QtCore from pyface.qt
#from pyface.qt import QtGui, QtCore

import mayavi.mlab as ML
import traits.api as TA
class Visualization(TA.HasTraits):
    import traitsui.api as TUA # Doing this import creates a QApplication
    import mayavi.core.ui.api as MCUA
    # Scene variable
    scene = TA.Instance(MCUA.MlabSceneModel, ())
    # The panel layout
    view = TUA.View(
        TUA.Item('scene', editor=MCUA.SceneEditor(), resizable=True,
                 show_label=False),
        resizable=True,
        )
    @TA.on_trait_change('scene.activated')
    def create_pipeline(self):
        ''' Put frame/axes around surface plot
        '''
        ML.axes(ranges=self.ranges,xlabel='P',ylabel='v',zlabel='E')
        ML.outline()
    def __init__(self):
        """ Calculate three 2-d arrays of values to describe EOS
        surface and put the surface into self.scene
        """
        import numpy as np
        import eos
        EOS = eos.shaw()
        TA.HasTraits.__init__(self)
        P, v = np.mgrid[1e10:4e10:20j, 1e-6:4e-6:20j]
        P = P.T
        v = v.T
        E = np.empty(v.shape)
        n_i,n_j = v.shape
        for i in range(n_i):
            for j in range(n_j):
                E[i,j] = EOS.Pv2E(P[i,j],v[i,j])
        self.ranges = []
        for a in (P,v,E):
            self.ranges += [a.min(),a.max()]
        scale = lambda z: (z-z.min())/(z.max()-z.min())
        ML.mesh(scale(P),scale(v),scale(E),figure=self.scene.mayavi_scene)
        self.flag = False
#-----------------------------------------------------------------------------
# The QWidget containing the visualization
from PySide.QtGui import QWidget, QVBoxLayout
class MayaviQWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        self.visualization = Visualization()
        # The edit_traits call generates the widget to embed.
        self.ui = self.visualization.edit_traits(
            parent=self, kind='subpanel').control
        layout.addWidget(self.ui)
        self.ui.setParent(self)
