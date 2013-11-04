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

plot_sources = ['plot.py','calc.py']

gun=Environment()
#gun.PDF('ds13.tex') scons scanner fails.  Type pdflatex ds13 on CL
gun.PDF('notes.tex')
gun.PDF('juq.tex')
gun.Command(
    plot_targets,    # Targets
    plot_sources,    # Sources
    plot_command     # Command
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

# FixMe: Have you no shame?  Scons calls make here.
integral=Environment()
integral.Command(
    ('taylor.pdf', 'bounds_04.pdf'),
    ('integral.py',),
    'make taylor.pdf; make bounds_04.pdf; make bounds_005.pdf; make bounds_dg.pdf'
    )
#---------------
# Local Variables:
# eval: (python-mode)
# End:
