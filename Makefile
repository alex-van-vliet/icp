install:
	[ -d env ] || (python -m venv env; source env/bin/activate; pip install -r requirements.txt)

gen_graphs: install
	mkdir -p time_vs_version threshold_graphs bench_dataframes bench_comparisons
	(source env/bin/activate; python graph_generator.py)
