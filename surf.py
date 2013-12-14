'''
'''
import ideal_eos
import traits.api as TA
# To be able to use PySide or PyQt4 and not run in conflicts with traits,
# we need to import QtGui and QtCore from pyface.qt
#from pyface.qt import QtGui, QtCore
from PySide import QtGui, QtCore

EOS = ideal_eos.EOS()

import mayavi.mlab as ML
class Visualization(TA.HasTraits):
    import traitsui.api as TUA
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
        # Can't make axes or outline till scene is active
        ML.axes(ranges=self.ranges,xlabel='P',ylabel='v',zlabel='E')
        ML.outline()
        x,y,z = self.point_3d()
        self.point = ML.points3d(x,y,z,
                    color=(1,0,0),mode='2dcircle',scale_factor=.005)
    def __init__(self):
        """ Calculate three 2-d arrays of values to describe EOS
        surface and put the surface into self.scene
        """
        import numpy as np
        TA.HasTraits.__init__(self)
        P, v = np.mgrid[1e10:4e10:20j, 1e-6:4e-6:20j]
        P = P.T
        v = v.T
        E = EOS.Pv2E(P,v)
        self.ranges = []
        for a in (P,v,E):
            self.ranges += [a.min(),a.max()]
        scale = lambda z: (z-z.min())/(z.max()-z.min())
        ML.mesh(scale(P),scale(v),scale(E),figure=self.scene.mayavi_scene)
        self.flag = False
#-----------------------------------------------------------------------------
# The QWidget containing the visualization, this is pure PyQt4 code.
class MayaviQWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        layout = QtGui.QVBoxLayout(self)
        layout.setSpacing(0)
        self.visualization = Visualization()

        # If you want to debug, beware that you need to remove the Qt
        # input hook.
        #QtCore.pyqtRemoveInputHook()
        #import pdb ; pdb.set_trace()
        #QtCore.pyqtRestoreInputHook()

        # The edit_traits call will generate the widget to embed.
        self.ui = self.visualization.edit_traits(
            parent=self, kind='subpanel').control
        layout.addWidget(self.ui)
        self.ui.setParent(self)
