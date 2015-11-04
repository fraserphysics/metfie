# For making documents that describe fitting an eos.  Now the list of
# documents is:

# 1. notes.pdf
#    "rm *.pdf; time make notes.pdf" reports 0m51.413s on an intel
#    G3220 @ 3.00GHz cpu

# Use python2 rather than python3 because debian doesn't have a
# python3 cvxopt version.

# FIGURES is a list of all figures required for notes.pdf.  See these docs:
# https://www.gnu.org/software/make/manual/html_node/Automatic-Variables.html
# https://www.gnu.org/software/make/manual/html_node/Text-Functions.html
# Type "make test" to see the list of figures
FIG_ROOT = C_gun vt_gun BC_gun opt_result big_d fve_gun
FIGURES = $(patsubst %, figs/%.pdf, ${FIG_ROOT}) basis.pdf

CODE = plot.py fit.py eos.py gun.py

notes.pdf: notes.aux notes.bbl ${FIGURES}
	pdflatex notes
notes.aux: notes.tex ${FIGURES}
	pdflatex notes
notes.bbl: notes.aux local.bib
	bibtex notes
local.bib: ../local.bib
	ln -s $^ $@

figs/%.pdf: ${CODE}
	python plot.py --$* $*.pdf

basis.pdf: basis.py
	python basis.py $@
test:
	@echo FIGURES = ${FIGURES}
clean:
	rm -rf figs *.pdf *.pyc *.log *.aux

###---------------
### Local Variables:
### eval: (makefile-mode)
### eval: (setq ispell-personal-dictionary "./localdict")
### End:
