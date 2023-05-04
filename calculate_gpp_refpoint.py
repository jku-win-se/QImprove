# Calculates the global Pareto population and global Pareto front

# Join all Pareto Populations to global-global pareto population
import pathlib

import click
import pandas as pd


def dominates(first, other, obj=slice(None)):
    """Return true if each objective of *self* is not strictly worse than
    the corresponding objective of *other* and at least one objective is
    strictly better.

    :param obj: Slice indicating on which objectives the domination is
                tested. The default value is `slice(None)`, representing
                every objectives.
    """
    not_equal = False
    for self_wvalue, other_wvalue in zip(first, other):
        if self_wvalue < other_wvalue:
            not_equal = True
        elif self_wvalue > other_wvalue:
            return False
    return not_equal

ignore_suffix = [
    "_globalPareto.csv",
    "_HVrefpoint.csv",
    "_earliest_finish.csv",
    "-PI.csv",
    "-normPI.csv",
    "-DCI.csv"
]

@click.command()
@click.argument("circuit", # help="Provide the path to the circuit file.",
                nargs=1, type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path))
@click.option("-r", "--results-dir", default="./results", help="Specify where the results should go",
              type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path))
def run_all(circuit: pathlib.Path, results_dir: pathlib.Path):
    merge_gpps(circuit, results_dir)
    calculate_refpoint_and_earliest_time(circuit, results_dir)

def merge_gpps(circuit: pathlib.Path, results_dir: pathlib.Path):
    columns = ["overlap", "num_gates", "depth", "num_nonloc_gates", "num_parameters"]

    print("Merging GPPs")
    problem_prefix = circuit.stem
    print(f"Problem Identifier: {problem_prefix}")
    all_problem_csv_files = list(results_dir.glob(f"{problem_prefix}*seed*.csv"))
    all_problem_csv_files = [f for f in all_problem_csv_files if "logbook.csv" not in str(f)]

    for suffix in ignore_suffix:
        all_problem_csv_files = [f for f in all_problem_csv_files if not str(f).endswith(suffix)]

    if len(all_problem_csv_files) != 210:
        print("WARNING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("There should be 210 result csv files available, but we only have", len(all_problem_csv_files))

    dfs = []
    for file in all_problem_csv_files:
        # print("file", file)
        df = pd.read_csv(file)
        last_pop = df[df.ngen == df.ngen.max()].drop_duplicates()
        dfs.append(last_pop)


    merged = pd.concat(dfs)
    merged = merged[columns].drop_duplicates()

    nondom_set = []

    as_np = merged.to_numpy()
    as_np[:,0] = 1 - as_np[:,0]

    for ind in as_np:
        # print("New", ind)
        is_dominated = False
        dominates_one = False
        has_twin = False
        to_remove = []

        for i, pf_ind in enumerate(nondom_set):
            if not dominates_one and dominates(pf_ind, ind):
                # print("New is dominated by ", pf_ind)
                is_dominated = True
                break
            elif dominates(ind, pf_ind):
                # print("New one dominates", pf_ind)
                dominates_one = True
                to_remove.append(i)
            elif (ind == pf_ind).all(): # and is_similar(ind, pf_ind):
                # print("Already in there")
                has_twin = True
                break

        # print("Nondom Set so far", nondom_set)

        for i in reversed(to_remove):  # Remove the dominated hofer
            # print("Removing dominated", i, nondom_set[i])
            nondom_set.pop(i)
        if not is_dominated and not has_twin:
            # print("Adding")
            nondom_set.append(ind)

    print("Final PF")
    df = pd.DataFrame(nondom_set, columns=columns)
    df[columns[0]] = 1 - df[columns[0]]
    print(df)

    print("Writing to file", results_dir / f"{problem_prefix}_globalPareto.csv")
    df.to_csv(results_dir / f"{problem_prefix}.csv", index=False)

    # print(all_problem_csv_files)
    print("Total:", len(all_problem_csv_files), "files")


def calculate_refpoint_and_earliest_time(circuit: pathlib.Path, results_dir: pathlib.Path):
    columns = ["overlap", "num_gates", "depth", "num_nonloc_gates", "num_parameters"]

    print("Calculating Refpoint and earliest stop time")
    problem_prefix = circuit.stem
    print(f"Problem Identifier: {problem_prefix} (circuit: {circuit})")
    all_problem_csv_files = list(results_dir.glob(f"{problem_prefix}*seed*.csv"))
    all_problem_csv_files = [f for f in all_problem_csv_files if "logbook.csv" not in str(f)]

    for suffix in ignore_suffix:
        all_problem_csv_files = [f for f in all_problem_csv_files if not str(f).endswith(suffix)]

    print("Analysing", len(all_problem_csv_files), "files")
    if len(all_problem_csv_files) != 210:
        print("WARNING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("There should be 210 result csv files available, but we only have", len(all_problem_csv_files))

    dfs = []
    min_dfs = []
    max_dfs = []
    for file in all_problem_csv_files:
        # print("file", file)
        df = pd.read_csv(file)
        dfs.append(df.min())
        min_dfs.append(df.min())
        dfs.append(df.max())
        max_dfs.append(df.max())

        # print(file.stem, round(df.timestamp.max(),2))

    merged_df = pd.DataFrame(dfs)
    merged_df = merged_df[columns]
    min = merged_df.min()
    max = merged_df.max()

    ref = [0.0] + list(merged_df[columns[1:]].max().values + 1)
    print("HV Refpoint is", ref)
    # Write to file...
    print("Writing Refpoint to file", results_dir / f"{problem_prefix}_HVrefpoint.csv")
    pd.DataFrame([ref], columns=columns).to_csv(results_dir / f"{problem_prefix}_HVrefpoint.csv", index=False)

    max_vals = pd.DataFrame(pd.Series([df.timestamp.max() for df in max_dfs]))
    print("Analysis of execution times:", pd.DataFrame(max_vals.describe()).T)

    print("Writing Refpoint to file", results_dir / f"{problem_prefix}_earliest_finish.csv")
    pd.DataFrame(max_vals.describe()).T.to_csv(results_dir / f"{problem_prefix}_earliest_finish.csv", index=False)

if __name__ == "__main__":
    run_all()