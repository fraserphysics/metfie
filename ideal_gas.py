"""
This file is derived from  wx_embedding.py.  It is modified to display
an ideal gas EOS.

I will try to duplicate-improve-replace it using qt

http://github.enthought.com/traitsui/tutorials/index.html
http://github.enthought.com/traits/index.html
http://github.enthought.com/mayavi/mayavi/index.html
http://github.enthought.com/traitsui/traitsui_user_manual/factories_basic.html
"""
import support, numpy as np, mayavi.mlab as ML, traits.api as TA
import traitsui.api as TUA, mayavi.core.ui.api as MCUA

EOS = support.EOS()
# Values for initial slider positions:
P_0,v_0 = 3.0e10,3.0e-6
E_0 = EOS.Pv2E(P_0,v_0)
class MayaviView(TA.HasTraits):
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
# Wx Code
import wx

class MainWindow(wx.Frame):

    def __init__(self, parent, id):
        wx.Frame.__init__(self, parent, id,
                          'Equation of State via Mayavi and WX')
        self.mayavi_view = MayaviView()
        # Use traits to create a panel, and use it as the content of this
        # wx frame.
        self.control = self.mayavi_view.edit_traits(
                        parent=self,
                        kind='subpanel').control
        self.Show(True)

app = wx.PySimpleApp()
frame = MainWindow(None, wx.ID_ANY)
app.MainLoop()


#---------------
# Local Variables:
# eval: (python-mode)
# End:
