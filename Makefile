pdf_src = rapport
pdf_dest = rapport-GPGPU

pdf:
	pandoc $(pdf_src).md --standalone --toc --toc-depth 2 -f markdown --filter pandoc-include -o $(pdf_dest).pdf
