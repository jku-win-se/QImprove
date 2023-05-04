import numpy as np

GLOBAL_RNG = None


def seeded_rng(seed: int = None) -> np.random.Generator:
    """
    Once the seed is set, it is permanent, until you use `reset_rng` to reset.
    Thus, to set it, manually call it before any other function accesses.

    Args:
        seed (int): Which seed to use. If not, will use random seed.

    Returns:
        (np.random.Generator): The random number generator
    """
    global GLOBAL_RNG
    if GLOBAL_RNG is None:
        # logger.info(f"Setting random seed to {seed}")
        GLOBAL_RNG = np.random.default_rng(seed)
    return GLOBAL_RNG
