FIGURES = mt2.pdf bounds.pdf eigenfunctions.pdf eigenvalues.pdf
siads.pdf: siads.tex $(FIGURES)
	pdflatex siads
	pdflatex siads

bounds.pdf: plot.py
	python3 plot.py --bounds bounds.pdf
eigenfunctions.pdf: eigenvalues.pdf
	echo eigenfunctions.pdf depends on eigenvalues.pdf
eigenvalues.pdf: sym.py Makefile
	python sym.py --eigenvalues eigenvalues.pdf --n_calculate 11 \
--eigenfunctions eigenfunctions.pdf --ns 1 2 3 5 10

%.pdf %.pdf_t: %.fig  # Remember to use 'special' option for text in xfig
	fig2dev -L pdftex $*.fig $*.pdf
	fig2dev -L pdftex_t -p $*.pdf $*.fig $*.pdf_t

#---------------
# Local Variables:
# eval: (makefile-mode)
# End:
