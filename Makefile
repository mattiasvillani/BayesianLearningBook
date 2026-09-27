# Regenerate exercises/resolved-exercises/: a copy of the book's exercise .tex
# files with \ref/\eqref/\pageref replaced by the numbers from the book's .aux.
# Run automatically by the pre-render hook in _quarto.yml.

BOOK     := ../BayesBook
SCRIPT   := $(BOOK)/Scripts/resolve_crossrefs.py
AUX      := $(BOOK)/Text/BayesBook.aux
SRC      := $(BOOK)/Text/exercises
OUT      := exercises/resolved-exercises
STAMP    := $(OUT)/.stamp

AUX_ALL  := $(AUX) $(wildcard $(BOOK)/Text/chapters/*.aux)
TEX_ALL  := $(shell find $(SRC) -name '*.tex')

.PHONY: resolve clean-resolved

resolve: $(STAMP)

$(STAMP): $(SCRIPT) $(AUX_ALL) $(TEX_ALL)
	rm -rf $(OUT)
	python3 $(SCRIPT) --aux $(AUX) --src $(SRC) --out $(OUT) \
		--labels-json $(OUT)/labels.json
	touch $@

clean-resolved:
	rm -rf $(OUT)
