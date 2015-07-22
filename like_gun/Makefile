# Use python2 rather than python3 because debian doesn't have a
# python3 cvxopt version.

# rm *.pdf; time make notes.pdf on orr yields:
#real	0m38.919s
#user	0m37.468s
#sys	0m0.268s

fig_d.pdf: calc.py   # Makes fig_d.pdf and fig_fve.pdf
	python calc.py
basis.pdf: basis.py
	python basis.py $@
D_test.pdf: calc.py
	python calc.py test
FIGURES = D_test.pdf basis.pdf fig_d.pdf
notes.aux: notes.tex ${FIGURES}
	pdflatex notes
notes.pdf: notes.aux ${FIGURES}
	pdflatex notes
#https://www.gnu.org/software/make/manual/html_node/Automatic-Variables.html
demo.tar: cmf_models.py markup.py calc.py
	tar -cf $@ $^

###---------------
### Local Variables:
### eval: (makefile-mode)
### eval: (setq ispell-personal-dictionary "./localdict")
### End:
