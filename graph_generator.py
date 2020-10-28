# %%
import pandas as pd
from pathlib import Path
import seaborn as sns
from IPython.display import display, HTML
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


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
        sns.scatterplot(data=data, x="threshold", y="cpu_time", hue="type", ax=ax[i])

    fig.savefig(f"threshold_graphs/threshold_vs_time_{data.iloc[0].bench}.png")

# %%
def time_vs_version(all_benches, min_version=None):
    fig, axs = plt.subplots(nrows=len(all_benches.test_id.unique()), figsize=(8, 20))
    plt.subplots_adjust(hspace=0.4)

    def time_vs_version_specialized(i, test_id, run_type, min_version=None):
        handle = [mpatches.Patch(color="red", label="GPU"), mpatches.Patch(color="blue", label="CPU")]
        axs[i].legend(handles=handle)

        color = "blue" if run_type == "BM_CPU" else "red"
        threshold = 32 if run_type == "BM_CPU" else 256

        bench_type = all_benches[(all_benches["type"] == run_type) & (all_benches["test_id"] == test_id)]
        if min_version is not None:
            bench_type = bench_type[bench_type["bench"] > min_version]
        bench_type = bench_type[(bench_type.threshold.isna()) | (bench_type.threshold == str(threshold))]
        bench_type = bench_type.sort_values("bench")

        sns.lineplot(data=bench_type, x="bench", y="cpu_time", ax=axs[i], color=color)
        axs[i].set_title(f"{run_type[-3:]}: {bench_type.iloc[0].label}")
        axs[i].set_xlabel("versions")
        axs[i].set_ylabel("time in ms")

    for i, test_id in enumerate(all_benches.test_id.unique()):
        time_vs_version_specialized(i, test_id, "BM_CPU", min_version)
        time_vs_version_specialized(i, test_id, "BM_GPU", min_version)

    fig.savefig(f"time_vs_version/time_vs_version-{min_version if min_version is not None else 'full'}.png")

def compare(bench):
    version = bench['bench'].unique()
    if len(version) != 1:
        print("too many versions")
        return
    version = version[0]
    comparison = pd.DataFrame(columns=['bench', 'threshold', 'test_id', 'label'])
    if 'threshold' in bench.columns:
        columns = ['test_id', 'threshold']
    else:
        columns = ['test_id']

    comparison[[*columns, 'label']] = bench[[*columns, 'label']].drop_duplicates()
    cpu_data = bench[bench['type'] == 'BM_CPU'][[*columns, 'cpu_time']].rename(columns={'cpu_time': 'cpu_time_cpu'})
    gpu_data = bench[bench['type'] == 'BM_GPU'][[*columns, 'cpu_time']].rename(columns={'cpu_time': 'cpu_time_gpu'})
    comparison = pd.merge(comparison, cpu_data, left_on=columns, right_on=columns, how='outer')
    comparison = pd.merge(comparison, gpu_data, left_on=columns, right_on=columns, how='outer')
    comparison['bench'] = version
    with open(f"bench_comparisons/{bench.iloc[0].bench}.md", "w") as file:
        file.write(comparison.to_markdown(tablefmt="grid"))

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
        csv = csv[["bench", "type", "test_id", "cpu_time", "label"]]
    elif len(tmp.columns) == 4:
        csv[["type", "threshold", "test_id", "useless"]] = tmp
        csv = csv[["bench", "type", "threshold", "test_id", "cpu_time", "label"]]
    else:
        import pdb; pdb.set_trace()
        print(f"Unhandled case: {bench_path.name}")

    threshold_vs_time(csv)
    compare(csv)
    with open(f"bench_dataframes/{csv.iloc[0].bench}.md", "w") as file:
        file.write(csv.to_markdown(tablefmt="grid"))

    return csv

# %%
def main():
    benchs = Path("benchmarks_logs")

    dfs = []

    for bench in benchs.glob("*.csv"):
        dfs.append(get_benchmark(bench))

    full_data = pd.concat(dfs)

    time_vs_version(full_data)
    time_vs_version(full_data, "v08")

# %%

if __name__ == '__main__':
    main()
