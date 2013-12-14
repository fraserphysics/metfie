"""This file is derived from wx_embedding.py.  It is modified to display
an ideal gas EOS.

To do:

1. Write mayavi code without traits
2. Move controls from traits to qt in this script
3. Move dot smoothly
4. Draw nice lines smoothly
5. Erase lines
6. Strip down

http://github.enthought.com/traitsui/tutorials/index.html
http://github.enthought.com/traits/index.html
http://github.enthought.com/mayavi/mayavi/index.html
http://github.enthought.com/traitsui/traitsui_user_manual/factories_basic.html

"""
# First, and before importing any Enthought packages, set the ETS_TOOLKIT
# environment variable to qt4, to tell Traits that we will use Qt.
import os
os.environ['ETS_TOOLKIT'] = 'qt4'
# To be able to use PySide or PyQt4 and not run in conflicts with traits,
# we need to import QtGui and QtCore from pyface.qt
from pyface.qt import QtGui, QtCore
# Alternatively, you can bypass this line, but you need to make sure that
# the following lines are executed before the import of PyQT:
#   import sip
#   sip.setapi('QString', 2)

from ideal_eos import EOS as IDEAL
import mayavi.mlab as ML
import traits.api as TA

EOS = IDEAL()
# Values for initial slider positions:
P_0,v_0 = 3.0e10,3.0e-6
E_0 = EOS.Pv2E(P_0,v_0)
class Visualization(TA.HasTraits):
    import traitsui.api as TUA
    import mayavi.core.ui.api as MCUA

    # Trait control vaiables:
    P = TA.Range(1.0e10,4.0e10,P_0)
    v = TA.Range(1.0e-6,4.0e-6,float(v_0))
    E = TA.Range(2.0e4,4.0e5,E_0)
    M = TA.Range(1.0e0,1.0e3,1.0e2)     # Mass of piston
    Constant = TA.Enum('P','v','E','S')
    Initial = TA.Button()
    Final = TA.Button()
    Integrate = TA.Button()
    # Scene variable
    scene = TA.Instance(MCUA.MlabSceneModel, ())
    # The panel layout
    view = TUA.View(
        TUA.Item('scene', editor=MCUA.SceneEditor(), resizable=True,
                 show_label=False),
        TUA.VGroup('_','P','v','E','M'),
        TUA.Item('Constant',style='custom',editor=TUA.EnumEditor(cols=3,
        values={'P':'Pressure','v':'volume','E':'energy','S':'Entropy'})),
        TUA.Item('Initial',show_label=False),
        TUA.Item('Final',show_label=False),
        TUA.Item('Integrate',show_label=False),
        resizable=True,
        )
    def _Initial_fired(self):
        self.P_i = self.P
        self.v_i = self.v
    def _Final_fired(self):
        self.v_f = self.v
    def _Integrate_fired(self):
        KE,t,KE_A = EOS.isentropic_expansion(
            self.P_i,self.v_i,self.v_f,mass=self.M)
        print("""
 %e Joules analytic result
 %e Joules via numerical integration
 %e seconds via numerical integration"""%(KE_A,KE,t))
    def point_3d(self):
        # Makes triple of numpy arrays from slider positions for plotting
        import numpy as np
        scale = lambda z,i: np.array([(z-self.ranges[2*i])/(
            self.ranges[2*i+1]-self.ranges[2*i])],np.float64)
        return (scale(self.P,0),scale(self.v,1),scale(self.E,2))
    # Service methods to respond to control changes
    @TA.on_trait_change('P')
    def change_P(self):
        if self.flag:
            return
        self.flag = True
        if self.Constant == 'v':
            self.E = EOS.Pv2E(self.P,self.v)
        if self.Constant == 'E':
            self.v = EOS.PE2v(self.P,self.E)
        if self.Constant == 'S':
            self.v,self.E = EOS.isentrope_P(self.P,self.v,self.E)
        self.flag = False
    @TA.on_trait_change('E')
    def change_E(self):
        if self.flag:
            return
        self.flag = True
        if self.Constant == 'v':
            self.P = EOS.Ev2P(self.E,self.v)
        if self.Constant == 'P':
            self.v = EOS.PE2v(self.P,self.E)
        if self.Constant == 'S':
            self.P,self.v = EOS.isentrope_E(self.E,self.P,self.v)
        self.flag = False
    @TA.on_trait_change('v')
    def change_v(self):
        if self.flag:
            return
        self.flag = True
        if self.Constant == 'E':
            self.P = EOS.Ev2P(self.E,self.v)
        if self.Constant == 'P':
            self.E = EOS.Pv2E(self.P,self.v)
        if self.Constant == 'S':
            self.P,self.E = EOS.isentrope_v(self.v,self.P,self.E)
        self.flag = False
    @TA.on_trait_change('P,E,v')
    def plot_PEv(self):
        x,y,z = self.point_3d()
        #self.point.mlab_source.set(x=x,y=y,z=z,scalars=z)
        ML.points3d(x,y,z,color=(1,0,0),mode='2dcircle',scale_factor=.005)
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


def main():
    # Don't create a new QApplication, it would unhook the Events
    # set by Traits on the existing QApplication. Simply use the
    # '.instance()' method to retrieve the existing one.
    app = QtGui.QApplication.instance()
    container = QtGui.QWidget()
    container.setWindowTitle("Embedding Mayavi in a PyQt4 Application")
    # define a "complex" layout to test the behaviour
    layout = QtGui.QGridLayout(container)

    # put some stuff around mayavi
    label_list = []
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 2:
                continue
            if (i==1) and (j==1):continue
            label = QtGui.QLabel(container)
            label.setText("Your QWidget at (%d, %d)" % (i,j))
            label.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
            layout.addWidget(label, i, j)
            label_list.append(label)
    mayavi_widget = MayaviQWidget(container)
    slider=QtGui.QSlider(QtCore.Qt.Vertical, container)
    slider.setRange(0,127)
    slider.setFixedWidth(30)
    layout.addWidget(slider, 1, 2)
    layout.addWidget(mayavi_widget, 1, 1)
    container.show()
    window = QtGui.QMainWindow()
    window.setCentralWidget(container)
    window.show()

    # Start the main event loop.
    app.exec_()

if __name__ == "__main__":
    main()
#---------------
# Local Variables:
# eval: (python-mode)
# End:
