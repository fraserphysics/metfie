# Use python2 rather than python3 because debian doesn't have a
# python3 cvxopt version.

# rm *.pdf; time make notes.pdf reports 0m51.413s on an intel G3220 @ 3.00GHz cpu

# FIGURES is a list of all figures required for notes.pdf.  See these docs:
#https://www.gnu.org/software/make/manual/html_node/Automatic-Variables.html
#https://www.gnu.org/software/make/manual/html_node/Text-Functions.html
FIG_ROOT = D_gun vt_gun BD_gun opt_result big_d fve_gun
FIGURES = $(patsubst %, figs/%.pdf, ${FIG_ROOT}) basis.pdf

figs/%.pdf: plot.py
	python plot.py --$* $*.pdf

basis.pdf: basis.py
	python basis.py $@
notes.aux: notes.tex ${FIGURES}
	pdflatex notes
notes.pdf: notes.aux ${FIGURES}
	pdflatex notes

###---------------
### Local Variables:
### eval: (makefile-mode)
### eval: (setq ispell-personal-dictionary "./localdict")
### End:
