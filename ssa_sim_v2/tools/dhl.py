# Fix paths for imports to work in unit tests ----------------

if __name__ == "__main__":
    
    from _fix_paths import fix_paths
    fix_paths()

# ------------------------------------------------------------

# Load libraries ---------------------------------------------

import numpy as np

# ------------------------------------------------------------


def randargmax(a, rng=None, max_fn=np.max):
    """
    Returns index of max element, with random tie breaking.
    If random tie breaking is not needed, use np.argmax() instead.

    :param Union[list, np.ndarray] a: A list or np.ndarray.
    :param object rng: Random number generator. If None, np.random is used.
        Must implement the choice method which chooses a random element of an array.
    :param function max_fn: A function choosing the maximal element.
    :return: Index of the max element.
    :rtype: int
    """

    temp = np.flatnonzero(a == max_fn(a))
    try:
        if rng is None:
            return np.random.choice(temp)
        else:
            return rng.choice(temp)
    except ValueError:
        print('DEBUG: set breakpoint here in dhl.randargmax()')
        return np.nan

def randargmax_ignoreNaN(a, rng=None):
    """
    Returns index of max element, with random tie breaking.
    If random tie breaking is not needed, use np.argmax() instead.
    Will return the index even if there are NaNs in the array.

    :param Union[list, np.ndarray] a: A list or np.ndarray.
    :param object rng: Random number generator. If None, np.random is used.
        Must implement the choice method which chooses a random element of an array.
    :param function max_fn: A function choosing the maximal element.
    :return: Index of the max element.
    :rtype: int
    """

    return randargmax(a, rng, max_fn=np.nanmax)


#
def find_nearest(val, arr):
    """
    Returns element, index in arr that is nearest to val.

    :param float val: Value to be approximated.
    :param Union[list, np.ndarray] arr: List or array with values.
    :return: Tuple of element and index in array nearest to val.
    :rtype: tuple
    """

    ix = np.argmin(np.abs(np.asarray(arr) - val))
    return arr[ix], ix


def avg_running(old_est, n_th_sample, n):
    """
    Calculates the running average.

    :param old_est: Previous value of the average.
    :param n_th_sample: New value to be included in the average.
    :param n: Overall number of samples.
    :return: Average including the new value.
    :rtype: float
    """
    assert(n > 0)
    return 1/n * n_th_sample + (n-1)/n * old_est
