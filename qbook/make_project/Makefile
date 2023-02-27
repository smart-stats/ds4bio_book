project.pdf : rplots.pdf project.tex
	    pdflatex project.tex

rplots.pdf : gen_rplots.R
	   R --no-save < gen_rplots.R


clean :
	rm -rf rplots.pdf
	rm -rf project.pdf
	rm -rf project.log
	rm -rf project.aux

