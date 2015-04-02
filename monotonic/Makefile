%.pdf %.pdf_t: %.fig  # Remember to use 'special' option for text in xfig
	fig2dev -L pdftex $*.fig $*.pdf
	fig2dev -L pdftex_t -p $*.pdf $*.fig $*.pdf_t

#---------------
# Local Variables:
# eval: (makefile-mode)
# End:
