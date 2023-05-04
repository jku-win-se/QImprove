import sys

import click
import pathlib
import numpy as np

import logging

from src.utils import seeded_rng

from io import StringIO

log_stream = StringIO()
logging.basicConfig(
     stream=log_stream,  # log to variable
     level=logging.INFO,
     format= '[%(asctime)s] %(levelname)s - %(message)s',
     datefmt='%H:%M:%S'
 )

handler = logging.StreamHandler(sys.stdout)
logging.getLogger("QRefactoring").addHandler(handler)  # log to sysout too

logger = logging.getLogger("QRefactoring")
logger.setLevel(logging.DEBUG)

from src import main, utils


@click.command()
@click.argument("circuit", # help="Provide the path to the circuit file.",
                nargs=1, type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path))
@click.argument("settings", # help="Provide the path to the settings file.",
                nargs=1, type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path))
@click.option("-s", "--seed", type=int, multiple=True, default=[np.random.choice(1000)])
@click.option("-b", "--buggy", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path))
@click.option("-r", "--results-dir", default="./results", help="Specify where the results should go",
              type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path))
@click.option("-N", "--population", type=int, help="override population settings")
@click.option("--ngen", type=int, help="override population settings")
@click.option("--pop_seed", type=int, help="override seeding for initial population")
@click.option("--circ_opt", type=str, multiple=True, help="override circuit optimization steps performed in search")
@click.option("--param_handling", type=str, help="set 'non-hyb' for NON-HYBRID and 'fixed' for fixed gate set")
def run_experiment(circuit: pathlib.Path, settings: pathlib.Path, seed: list[int], buggy: pathlib.Path, results_dir: pathlib.Path,
                   population: int, ngen: int, pop_seed: int, circ_opt: list[str], param_handling: str):
    assert circuit.suffix == ".qasm", f"Circuit needs to be a .qasm file, not {circuit}"
    assert settings.suffix == ".json", f"Settings needs to be a .json file, not {settings}"

    circ_opt = list(circ_opt)

    settings_override = {}
    if population:
        settings_override["N"] = population
    if ngen:
        settings_override["NGEN"] = ngen
    if not param_handling:
        if not ngen:
            settings_override["NGEN"] = 150
    if pop_seed:
        settings_override["init_pop"] = pop_seed
    if circ_opt:
        settings_override["opt_within"] = circ_opt
    if param_handling == "non-hyb":
        settings_override["use_numerical_optimizer"] = False
        if not ngen:
            settings_override["NGEN"] = 1600
        if not population:
            settings_override["N"] = 100
    if param_handling == "fixed":
        settings_override["gateset"] = "fixed"
        if not ngen:
            settings_override["NGEN"] = 1600
        if not population:
            settings_override["N"] = 100


    logger.info(f"{settings_override=}")

    for run_idx, s in enumerate(seed):
        logger.debug(f"Setting seed to {s}")
        np.random.seed(s)
        utils.seeded_rng(s)

        # TODO: Setup seed, add default for seed --> done
        logger.info(f"Launching run #{run_idx} with seed {s}")
        main.main(circuit, settings, buggy, results_dir, run_id=f"seed{s}", settings_override=settings_override, log_stream=log_stream)


if __name__ == "__main__":
    # import multiprocessing
    # pool = multiprocessing.Pool(5)
    run_experiment()