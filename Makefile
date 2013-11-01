ALL: IST_8_12.pdf Stat_8_12.pdf help.pdf QM_2_12.pdf gun.pdf
	touch $@

PCA.pdf: plot.py calc.py
	python plot.py --plot_PCA=$@
nominal.pdf: plot.py calc.py
	python plot.py --plot_nominal $@
T_study.pdf: plot.py calc.py
	python plot.py --plot_q1 $@
T_studyR.pdf: plot.py calc.py
	python plot.py --plot_q1R $@
q0_1.pdf: plot.py calc.py
	python plot.py --plot_q0_1 $@
ellipsoid.pdf: plot.py calc.py
	python plot.py --plot_ellipsoid $@ --Plane 0 1 0 6 30 31
moments.pdf: plot.py calc.py
	python plot.py --plot_moments $@
invariant.pdf: plot.py calc.py
	python plot.py --plot_invariant $@
allowedET.pdf: plot.py calc.py
	python plot.py --plot_allowedET allowedET.pdf
mean.pdf: plot.py calc.py
	python plot.py --plot_mean $@

# Figures for illustrating the integral equation
taylor.pdf: integral.py
	python integral.py --Dy 0.03 --taylor taylor.pdf
bounds_04.pdf: integral.py
	python integral.py --Dy 0.04 --bounds1 $@
bounds_005.pdf: integral.py
	python integral.py --Dy 0.005 --bounds1 $@
bounds_dg.pdf: integral.py
	python integral.py --Dy 0.005 --bounds2 $@
eigenfunctionB_dy_000032.pdf: explore.py
	python explore.py --dy 0.000032 --n_g 1000 --n_h 2000 --file $@
eigenfunction_dy_%.pdf: explore.py
	python explore.py --dy 0.$* --n_g 500 --n_h 1000 --file $@
STUDY = eigenfunction_dy_000064.pdf eigenfunction_dy_000032.pdf         \
eigenfunction_dy_000128.pdf eigenfunction_dy_000256.pdf                 \
eigenfunction_dy_000512.pdf eigenfunction_dy_001024.pdf bounds_04.pdf   \
bounds_005.pdf bounds_dg.pdf taylor.pdf
notes.pdf: notes.tex ${STUDY}
	pdflatex notes.tex
FIGS = PCA.pdf nominal.pdf T_study.pdf allowedET.pdf invariant.pdf	\
moments.pdf ellipsoid.pdf

IST_8_12.pdf: IST_8_12.tex mt2.pdf ${FIGS}
Stat_8_12.pdf: Stat_8_12.tex ${FIGS} T_studyR.pdf
help.pdf: help.tex mt2.pdf_t
QM_2_12.pdf: QM_2_12.tex ${FIGS} q0_1.pdf mean.pdf

gun.aux: gun.tex
	pdflatex gun
gun.bbl: local.bib gun.aux
	bibtex gun
gun.pdf: gun.tex gun.bbl ${FIGS}
ball.tar: gun.pdf gun.tex local.bib ${FIGS} Makefile plot.py calc.py
	tar -cf $@ $^
%.pdf: %.tex
	pdflatex $*

%.pdf %.pdf_t: %.fig  # Remember to use 'special' option for text in xfig
	fig2dev -L pdftex $*.fig $*.pdf
	fig2dev -L pdftex_t -p $*.pdf $*.fig $*.pdf_t

hixson.pdf lognom.pdf: shaw.py
	python shaw.py hixson.pdf lognom.pdf
ds13.pdf: ds13.tex ${FIGS} hixson.pdf lognom.pdf schematic.pdf

#---------------
# Local Variables:
# eval: (makefile-mode)
# End:
