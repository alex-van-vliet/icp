pdf_src = rapport
pdf_dest = rapport-GPGPU
template = https://raw.githubusercontent.com/Wandmalfarbe/pandoc-latex-template/master/eisvogel.tex

pdf:
	pandoc $(pdf_src).md --standalone --toc --toc-depth 2 -f markdown --template=$(template) -o $(pdf_dest).pdf

watch:
	(while inotifywait -e close_write *.md *.png; do make; done)
