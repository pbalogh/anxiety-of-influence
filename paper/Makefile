all: main.pdf

main.pdf: main.tex references.bib
	tectonic main.tex

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf

.PHONY: all clean
