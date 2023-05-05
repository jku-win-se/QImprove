# QImprovement

This repository provides the supporting material and source code for the analysis of search-based quantum program improvement.

This code depends on [DEAP](https://deap.readthedocs.io/) for the implementation of the search approach, 
and on [pymoo](https://pymoo.org/), which we used to compute IGD and HyperVolume.

## Repository Structure

* [`README.md`](README.md) this file.

### Supporting Material
* [`evaluation/supporting-material.pdf`](evaluation/supporting-material.pdf) A file containg data tables and supplementary data, which we could not fit into the paper. 

### Files and folders required to run experiments
* [`run_experiment.py`](run_experiment.py) entry point of a search run. Use via 
```
python run_experiment.py examples/some/qantumprogram.qasm examples/some/settings.json --results-dir ./results/folder --seed 1234 [otheroptions]
```
* [`experiments.csv`](experiments.csv) a list of quantum experiments that we performed.
* [`options.txt`](options.txt) each experiment was executed using these seven options.
* [`seeds.txt`](seeds.txt) the random seeds for the 30 experiment repetitions.
* [`examples`](examples) contains a set of quantum programs (.qasm) and settings (.json) files
* [`src`](src) contains source for the search, etc.
* [`requirements.txt`](requirements.txt) the Pypi dependencies of the source code

### Files and folders required to analyse the data:
* [`calculate_gpp_refpoint.py`](calculate_gpp_refpoint.py) Calculates the global Pareto population (i.e. the best of all problems) and the Reference Point for a HyperVolume (i.e the worst dimensional values + 1) for a problem and stores it in a file.
```
python calculate_gpp_refpoint.py examples/some/quantumprogram.qasm --results-dir ./results/folder
```
* [`run_analysis.py`](run_analysis.py) Calculates performance indicators (e.g. GD / IGD / HV) and DCI for a given experiment run.
Call the script on an output file, like so:
```
python run_analysis.py ./results/folder/qantumprogram_seed1234_optionA.csv
```
* [`analysis`](analysis) contains the Jupyter notebooks for the data analysis, as well as the merged data used for the analysis in the paper.
  (Raw data is too large for Github, but available upon request.)

## How we ran the experiments

The data for these experiments was produced using a computation cluster provided by our institution.
Thus, there is no "single make file", which triggers the entire computation.
Instead, the cluster uses [SLURM](https://slurm.schedmd.com/) as a job submission interface.

We therefore had to execute the following steps in order:
1. Create a `run_experiment.py`-job for each combination of `experiment x option x seed` (in files: `experiments.csv`, `options.txt`, `seeds.txt`)
2. Run the `calculate_gpp_refpoint.py` for each experiment (in file `experiments.csv`)
3. Execute `run_analysis.py` on each csv-output file produced in Step 1.

Finally, we could download the data and analyse it locally.


# Thanks

> Our gratitude goes to the computation cluster team at Johannes Kepler University Linz for their help and support in executing our experiments.


> Thanks also to Jackson Antonio do Prado Lima [(@jacksonpradolima)](https://gist.github.com/jacksonpradolima) for implementing and publishing
his Python-port of the [VD_A analysis](https://gist.github.com/jacksonpradolima/f9b19d65b7f16603c837024d5f8c8a65)