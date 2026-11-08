LATEX = pdflatex
TARGET = report_vi
CHROMIUM = chromium

.PHONY: all html-pdf clean

all:
	$(LATEX) -interaction=nonstopmode -halt-on-error $(TARGET).tex
	$(LATEX) -interaction=nonstopmode -halt-on-error $(TARGET).tex

# Portable fallback when Vietnamese TeX fonts are unavailable locally.
html-pdf:
	pandoc $(TARGET).tex --from=latex --to=html5 --standalone --toc \
		--toc-depth=2 --mathml --embed-resources --resource-path=. \
		--css=report_print.css --include-after-body=report_print.js \
		--metadata pagetitle='Báo cáo Bài tập lớn Học máy' -o $(TARGET).html
	$(CHROMIUM) --headless --no-sandbox --disable-gpu \
		--user-data-dir=/tmp/ml-report-chromium --no-pdf-header-footer \
		--print-to-pdf=$(TARGET).pdf file://$(CURDIR)/$(TARGET).html

clean:
	rm -f $(TARGET).aux $(TARGET).log $(TARGET).out $(TARGET).toc \
	      $(TARGET).lof $(TARGET).lot $(TARGET).fls $(TARGET).fdb_latexmk \
	      $(TARGET).html
