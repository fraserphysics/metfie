'''SConstruct: For exploring the scons build system.

SCons User Guide at
http://www.scons.org/doc/production/HTML/scons-user/index.html

Wiki at
www.scons.org/wiki

Documentation of LaTeX scanner class at
http://www.scons.org/doc/HTML/scons-api/SCons.Scanner.LaTeX.LaTeX-class.html

http://www.scons.org/wiki/LatexSupport
'''

def build_pdf_t(target, source, env):
    ''' Written for the fig2pdf Builder, this function runs fig2dev
    twice on an x-fig source.
    "target" is two Nodes [*.pdf, *.pdf_t]
    "source" is single Node [*.fig]
    '''
    import subprocess
    x_fig = str(source[0])
    x_pdf = str(target[0])
    x_pdf_t = str(target[1])
    subprocess.call(['fig2dev','-L','pdftex',x_fig,x_pdf])
    subprocess.call(['fig2dev', '-L', 'pdftex_t', '-p', x_pdf, x_fig, x_pdf_t])
    return None

'''fig2pdf is a SCons Builder for making "%.pdf" and "%.pdf_t" from
"%.fig".  The arguments of the emitter function, "target" and
"source", are lists of SCons Nodes.'''
fig2pdf = Builder(
    action=build_pdf_t, src_suffix='.fig', suffix='.pdf',
    emitter=lambda target,source,env:([target[0],str(target[0])+'_t'],source))

# For standard gun, collect flags and targets for single invocation of plot.py.
plot_targets = []
plot_command = 'python3 plot.py --Plane 0 1 0 6 30 31'
for flag,target in (('--plot_%s'%root,'%s.pdf'%root) for root in(
    'PCA','nominal','q0_1','moments','invariant','T_study','allowedET',
    'ellipsoid')):
    plot_command += ' %s %s'%(flag,target)
    plot_targets.append(target)

plot_sources = ['plot.py','calc.py', 'C.cpython-33m.so']

gun=Environment()
#gun.PDF('ds13.tex') scons scanner fails.  Type pdflatex ds13 on CL
gun.PDF('juq.tex')
gun.Command(
    plot_targets,    # Targets
    plot_sources,    # Sources
    plot_command     # Command
    )
gun.Command(
    ('C.cpython-33m.so', 'first_c.cpython-33m.so'),
    ('C.pyx', 'first_c.pyx'),
    'python3 setup.py build_ext --inplace'
    )
gun.Command(
    'allowed_tp2.pdf',
    'plot.py',
    'python3 plot.py --plot_allowed_tp2 allowed_tp2.pdf --N 600 --Nf 50'
    )
gun.Command(
    'pert.pdf',
    'plot.py',
    'python3 plot.py --plot_pert pert.pdf --N 300 --Nf 50'
    ) #  --N 500 --Nf 10 Just to save time
gun.Command(
    tuple(
    'recursive%s.pdf'%s for s in ('I', 'II', 'III', 'IV', 'V')),
    'recursive.py',
    'python3 recursive.py'
    )

ist = gun.Clone()
ist.PDF('IST_8_12.tex')
ist['BUILDERS']['Fig'] = fig2pdf
ist.Fig('mt2')

# Need eos.pdf and isentropePv.pdf from ideal_gas.py GUI
stat = ist.Clone()
stat.PDF('Stat_8_12.tex')
stat.Command('T_studyR.pdf',plot_sources,
             'python3 plot.py --plot_q1R T_studyR.pdf')

Help = ist.Clone()
Help.PDF('help.tex')

qm = gun.Clone()
qm.PDF('QM_2_12.tex')
qm.Command('mean.pdf',plot_sources,'python3 plot.py --plot_mean mean.pdf')

sources = ('calc.py', 'MaxH.py', 'plot.py', 'SConstruct', 'juq.tex', 'juq.pdf',
           'notes.tex', 'notes.pdf', 'local.bib', 'README')
eos=Environment()
eos.Command(
    ('metfie.tar',),
    sources,
    'tar -cf metfie.tar ' + (len(sources)*' %s ')%sources
    )

# Plots and tex for notes.pdf
notes=Environment()
notes.PDF('notes.tex')
notes.Command(
    ('taylor.pdf',),
    ('integral.py',),
    'python3 integral.py --Dy 0.03 --taylor taylor.pdf'
    )
notes.Command(
    ('bounds_04.pdf',),
    ('integral.py',),
    'python3 integral.py --Dy 0.04 --bounds1 bounds_04.pdf'
    )
notes.Command(
    ('bounds_005.pdf',),
    ('integral.py',),
    'python3 integral.py --Dy 0.005 --bounds1 bounds_005.pdf'
    )
notes.Command(
    ('bounds_dg.pdf',),
    ('integral.py',),
    'python3 integral.py --Dy 0.005 --bounds2 bounds_dg.pdf'
    )
notes.Command(
    ('eigenfunction.pdf', 'Av.pdf'),
    ('explore.py', 'first_c.cpython-33m.so'),
    'python3 explore.py --dy 5e-5 --n_g 400 --n_h 400 --eigenfunction eigenfunction.pdf --Av Av.pdf'
    )
u = 48.0
dy = 0.3
points_ = (
          -0.9 , 0.25,
           -0.4, -0.8,
              0, 0,
             .5, 0.9,
          0.995, 0.0)
points = (len(points_)*'%.3f ')%points_
notes.Command(
    ('Av1.pdf'),
    ('map2d.py', 'first_c.cpython-33m.so'),
    'python3 map2d.py --u %f --dy %f --out Av1.pdf --points %s'%(
        u, dy, points)
    )
m,b = 1/dy, u/dy - 6*dy
notes.Command(
    ('Av2.pdf'),
    ('map2d.py', 'first_c.cpython-33m.so'),
    'python3 map2d.py --u %f --dy %f --out Av2.pdf --backward --line %f %f --points %s'%(u, dy, m,b,points)
    )
dy = .05
m,b = 1/dy, u/dy - 6*dy
points_ = points_[2:] + (-.92, 0.99, -.995, 0.99)
points = (len(points_)*'%.3f ')%points_
notes.Command(
    ('Av3.pdf'),
    ('map2d.py', 'first_c.cpython-33m.so'),
    'python3 map2d.py --u %f --dy %f --out Av3.pdf --backward --line %f %f --points %s'%(u, dy, m,b,points)
    )

n_g = 800
n_h = 400
target_ = 'study_%d_%d_%s'
command_ = 'python3 converge.py --out_file %s --n_g0 200 --n_g_step 5 --n_g_final %d --n_h0 200 --n_h_step 4 --n_h_final %d --ref_frac .8 --dy 0.000%s'
for dy in ('64', '32', '16', '08'):
    target = target_%(n_g, n_h, dy)
    notes.Command(
        (target,),
        ('converge.py',),
        command_%(target, n_g, n_h, dy)
    )

import os.path
n_g = 5000
n_h = 1200
target_ = '%d_g_%d_h_%s_y_tol_1'
command_ = 'python3 archive.py --out_file %s --n_g %d --n_h %d  --dy 0.000%s --out_dir reference  --tol 1e-6'
for dy in ('64', '32', '16', '08', '04'):
    target = target_%(n_g, n_h, dy)
    notes.Command(
        (os.path.join('reference',target),),
        ('first_c.cpython-33m.so',),
        command_%(target, n_g, n_h, dy)
    )
quadfirst = notes.Clone()
quadfirst.Command(
    ('eric.pdf', 'eric.latex'),
    ('eric.py',),
    'python eric.py --eric eric.pdf --latex eric.latex'
    )
quadfirst.Command(
    ('f_n.pdf'),
    ('eric.py',),
    'python eric.py --f_n f_n.pdf'
    )
quadfirst.PDF('quadfirst.tex')
n_g = 2000
n_h = 800
target_ = '%d_g_%d_h_%s_y'
archive_command = 'python3 archive.py --out_file %s --n_g %d --n_h %d  --dy %s'
fig_command = (
'export DISPLAY=:0.0;python view.py --log_floor 1e-45 --resolution 200 200 '+
'--file %s --fig_files %s')
for dy in ('.4', '.2', '.1', '.05'):
    target = target_%(n_g, n_h, dy)
    archive_target = os.path.join('archive',target)
    fig_pattern = os.path.join('figs',target_%(n_g, n_h, dy[1:]))
    quadfirst.Command(
        (archive_target,),
        ('first_c.cpython-33m.so',),
        archive_command%(target, n_g, n_h, dy)
    )
    quadfirst.Command(
        (fig_pattern+'_vec_log.png',),
        tuple([]),
        fig_command%(target, fig_pattern)
    )
quadfirst.Command(
    'compare.pdf',
    [], #'archive/2000_g_800_h_.2_y',
    'python analysis.py --file 2000_g_800_h_.2_y --out compare.pdf')
# From command line "designer-qt4 PVE_control.ui"
qt = Environment()
qt.Command(
    ('ui_eos_qt.py',),
    ('eos_qt.ui', 'patch_ideal'),
    'pyside-uic eos_qt.ui > ui_eos_qt.py; patch ui_eos_qt.py patch_ideal'
    )
qt.Command(
    ('ui_PVE_control.py',),
    ('PVE_control.ui',),
    'pyside-uic PVE_control.ui > ui_PVE_control.py'
    )

#---------------
# Local Variables:
# eval: (python-mode)
# End:
