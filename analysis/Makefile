gen_graphs: install
	mkdir -p results
	(source env/bin/activate; python graph_generator.py)

borders: install
	(source env/bin/activate; python cleanborder.py)

install:
	[ -d env ] || (python -m venv env; source env/bin/activate; pip install -r requirements.txt)

clean:
	$(RM) results
