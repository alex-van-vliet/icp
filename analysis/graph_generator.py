# %%
import os
import pandas as pd
from pathlib import Path
import seaborn as sns
from IPython.display import display, HTML
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

def mkopen(path, *args, **kwargs):
    if isinstance(path, str):
        path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    return open(path, *args, **kwargs)

def get_figsize(x_cm, y_cm=None, nb_x=1, nb_y=1):
    x_inches = x_cm * 10 / 25.4
    y_inches = y_cm * 10 / 25.4 if y_cm else x_inches / 1920 * 1080
    x_inches *= nb_x
    y_inches *= nb_y
    return {'figsize': (x_inches,y_inches)}

# %%
def threshold_vs_time(bench):
    if "threshold" not in bench.columns:
        print(f"no treshold: {bench.iloc[0].bench}")
        return

    fig, ax = plt.subplots(figsize=(12, 30), nrows=5, ncols=1)

    for i, test in enumerate(bench.test_id.unique()):
        data = bench[bench.test_id == test]
        ax[i].set_title(data.iloc[0].label)
        ax[i].set_xlabel("VPTree threshold")
        ax[i].set_ylabel("time in ms")
        sns.scatterplot(data=data, x="threshold", y="real_time", hue="type", ax=ax[i])

    with mkopen(f"results/threshold_graphs/threshold_vs_time_{data.iloc[0].bench}.png", "wb") as file:
        fig.savefig(file)

# %%
def time_vs_version(all_benches, min_version=None):
    fig, axs = plt.subplots(nrows=len(all_benches.test_id.unique()), figsize=(8, 20))
    plt.subplots_adjust(hspace=0.4)

    def time_vs_version_specialized(i, test_id, min_version=None):
        bench_type = all_benches[all_benches["test_id"] == test_id]
        if min_version is not None:
            bench_type = bench_type[bench_type["bench"] > min_version]
        bench_type = bench_type.groupby(by=['test_id', 'label', 'bench', 'type'], dropna=False).min('real_time')
        bench_type = bench_type.reset_index()
        bench_type = bench_type.sort_values("bench")

        sns.lineplot(data=bench_type, x="bench", y="real_time", hue='type', ax=axs[i])
        axs[i].set_title(f"{bench_type.iloc[0].label}")
        axs[i].set_xlabel("versions")
        axs[i].set_ylabel("time in ms")

    for i, test_id in enumerate(all_benches.test_id.unique()):
        time_vs_version_specialized(i, test_id, min_version)

    with mkopen(f"results/time_vs_version/time_vs_version-{min_version if min_version is not None else 'full'}.png", "wb") as file:
        fig.savefig(file)

def compare_best(bench):
    if 'threshold' not in bench.columns:
        print("no theshold")
        return
    columns = ['bench', 'type', 'test_id', 'label']
    best_bench = bench[[*columns, 'real_time']].groupby(by=columns)['real_time'].idxmin()
    bench = bench.loc[best_bench].reset_index()

    version = bench['bench'].unique()
    if len(version) != 1:
        print("too many versions")
        return
    version = version[0]

    columns = ['test_id']
    comparison = pd.DataFrame(columns=['bench', *columns, 'label'])
    comparison[[*columns, 'label']] = bench[[*columns, 'label']].drop_duplicates()
    cpu_data = bench[bench['type'] == 'BM_CPU'][[*columns, 'real_time', 'threshold']]
    cpu_data = cpu_data.rename(columns={'real_time': 'real_time_cpu', 'threshold': 'threshold_cpu'})
    gpu_data = bench[bench['type'] == 'BM_GPU'][[*columns, 'real_time', 'threshold']]
    gpu_data = gpu_data.rename(columns={'real_time': 'real_time_gpu', 'threshold': 'threshold_gpu'})
    comparison = pd.merge(comparison, cpu_data, left_on=columns, right_on=columns, how='outer')
    comparison = pd.merge(comparison, gpu_data, left_on=columns, right_on=columns, how='outer')
    comparison['bench'] = version
    comparison = comparison.drop(columns=['test_id'])
    with mkopen(f"results/comparisons/{bench.iloc[0].bench}-best.md", "w") as file:
        file.write(comparison.to_markdown(index=False, tablefmt="github"))


def compare(bench):
    version = bench['bench'].unique()
    if len(version) != 1:
        print("too many versions")
        return
    version = version[0]
    if 'threshold' in bench.columns:
        columns = ['test_id', 'threshold']
    else:
        columns = ['test_id']

    comparison = pd.DataFrame(columns=['bench', *columns, 'label'])
    comparison[[*columns, 'label']] = bench[[*columns, 'label']].drop_duplicates()
    cpu_data = bench[bench['type'] == 'BM_CPU'][[*columns, 'real_time']].rename(columns={'real_time': 'real_time_cpu'})
    gpu_data = bench[bench['type'] == 'BM_GPU'][[*columns, 'real_time']].rename(columns={'real_time': 'real_time_gpu'})
    comparison = pd.merge(comparison, cpu_data, left_on=columns, right_on=columns, how='outer')
    comparison = pd.merge(comparison, gpu_data, left_on=columns, right_on=columns, how='outer')
    comparison['bench'] = version
    comparison = comparison.drop(columns=['test_id'])
    with mkopen(f"results/comparisons/{bench.iloc[0].bench}.md", "w") as file:
        file.write(comparison.to_markdown(index=False, tablefmt="github"))

def compare_graph(all_benches, versions=None, best=False):
    fig, ax = plt.subplots(**get_figsize(50), nrows=1, ncols=1)
    if not versions:
        versions = all_benches['bench'].unique()
    benches = all_benches[all_benches['bench'].isin(versions)].copy()
    if best:
        benches = benches.groupby(by=['bench', 'type', 'test_id', 'label']).min('real_time')
        benches = benches.reset_index()
    def name(row):
        method = {'BM_CPU': 'cpu', 'BM_GPU': 'gpu'}[row['type']]
        bench = row['bench']
        if 'threshold' in row and not row.isna()['threshold']:
            threshold = row['threshold']
            return f"{bench}-{threshold}-{method}"
        return f"{bench}-{method}"
    def order(name):
        t = name.split('-')
        if len(t) == 2:
            return (*t,)
        return (t[0], int(t[1]), t[2])
    benches['type'] = benches.apply(name, axis=1)
    order = sorted(benches['type'].unique(), key=order)
    sns.barplot(x="label", y="real_time", hue='type', hue_order=order, data=benches, ci=None)
    with mkopen(f"results/comparisons_graphs/{'-'.join(versions)}{'-best' if best else ''}.png", "wb") as file:
        fig.savefig(file)

# %%
def get_benchmark(bench_path):
    csv = pd.read_csv(bench_path)

    tmp = csv["name"].str.split("/", expand=True)
    bench_name = bench_path.name.split(".")[0].split("-")[0]
    if len(bench_name) == 2:
        bench_name = bench_name[:1] + "0" + bench_name[1:]
    csv["bench"] = bench_name

    if len(tmp.columns) == 3:
        csv[["type", "test_id", "useless"]] = tmp
        csv = csv[["bench", "type", "test_id", "real_time", "label"]]
    elif len(tmp.columns) == 4:
        csv[["type", "threshold", "test_id", "useless"]] = tmp
        csv = csv[["bench", "type", "threshold", "test_id", "real_time", "label"]]
    else:
        import pdb; pdb.set_trace()
        print(f"Unhandled case: {bench_path.name}")

    threshold_vs_time(csv)
    compare(csv)
    compare_best(csv)
    compare_graph(csv)
    with mkopen(f"results/dataframes/{csv.iloc[0].bench}.md", "w") as file:
        file.write(csv.to_markdown(tablefmt="grid"))

    return csv

# %%
def main():
    sns.set_theme()

    benchs = Path("logs")

    dfs = []

    for bench in benchs.glob("*.csv"):
        dfs.append(get_benchmark(bench))

    full_data = pd.concat(dfs)

    compare_graph(full_data, ['v01', 'v16'], best=True)

    time_vs_version(full_data)
    time_vs_version(full_data, "v08")

# %%

if __name__ == '__main__':
    main()
