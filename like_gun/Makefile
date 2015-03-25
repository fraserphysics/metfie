fig_d.pdf: calc.py
	python3 calc.py
notes.aux: notes.tex fig_d.pdf
	pdflatex notes
notes.pdf: notes.aux
	pdflatex notes
#https://www.gnu.org/software/make/manual/html_node/Automatic-Variables.html
demo.tar: cmf_models.py markup.py calc.py
	tar -cf $@ $^

###---------------
### Local Variables:
### eval: (makefile-mode)
### eval: (setq ispell-personal-dictionary "./localdict")
### End:
