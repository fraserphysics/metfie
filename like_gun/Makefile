fig_d.pdf: calc.py   # Makes fig_d.pdf and fig_fve.pdf
	python calc.py
basis.pdf: basis.py
	python basis.py $@
D_test.pdf: calc.py
	python calc.py test
FIGURES = D_test.pdf basis.pdf fig_d.pdf basis.pdf
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
