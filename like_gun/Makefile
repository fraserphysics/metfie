# Use python2 rather than python3 because debian doesn't have a
# python3 cvxopt version.

# rm *.pdf; time make notes.pdf on orr yields:
#real	0m38.919s
#user	0m37.468s
#sys	0m0.268s

basis.pdf: basis.py
	python basis.py $@
D_test.pdf: calc.py
	python calc.py test
d_hat_test.pdf errors_test.pdf opt_result.pdf vt_test.pdf: D_test.pdf
	echo making $^ made $@ too
fig_d.pdf: calc.py   # Makes fig_d.pdf and fig_fve.pdf
	python calc.py
fig_fve.pdf: fig_d.pdf
	echo making $^ made $@ too
BD_test.pdf: calc.py
	python calc.py BD
FIGURES = basis.pdf vt_test.pdf D_test.pdf BD_test.pdf opt_result.pdf	\
fig_d.pdf fig_fve.pdf
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
