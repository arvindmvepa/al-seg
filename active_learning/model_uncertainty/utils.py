from scipy.stats import entropy as entropy_func
import multiprocessing
import numpy as np
import sys


def entropy_w_label_probs(im_labels):
    im_labels = np.concatenate(im_labels)
    print(f"im_labels.shape: {im_labels.shape}")
    sys.stdout.flush()
    entropy_arr = parallel_apply_along_axis(pixel_entropy_w_probs, 0, im_labels)
    mean_entropy = np.mean(entropy_arr)
    return mean_entropy


def entropy_w_label_counts(im_labels):
    entropy_arr = parallel_apply_along_axis(pixel_entropy_w_label_counts, 0, im_labels)
    mean_entropy = np.mean(entropy_arr)
    return mean_entropy


def pixel_entropy_w_probs(pixel_probs):
    # take the average probability for each class and calculate entropy from that
    mean_per_model_pixel_probs = np.mean(pixel_probs, axis=0)
    return entropy_func(mean_per_model_pixel_probs)

def pixel_entropy_w_label_counts(pixel_labels):
    """https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python"""
    value,counts = np.unique(pixel_labels, return_counts=True)
    return entropy_func(counts)


def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    https://stackoverflow.com/questions/45526700/easy-parallelization-of-numpy-apply-along-axis
    Like numpy.apply_along_axis(), but takes advantage of multiple cores.
    """
    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    # Chunks for the mapping (only a few chunks):
    chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, multiprocessing.cpu_count())]

    pool = multiprocessing.Pool()
    print(f"chunks: {chunks}")
    individual_results = pool.map(unpacking_apply_along_axis, chunks)
    # Freeing the workers:
    pool.close()
    pool.join()

    return np.concatenate(individual_results)


def unpacking_apply_along_axis(all_args):
    """
    Like numpy.apply_along_axis(), but with arguments in a tuple
    instead.
    This function is useful with multiprocessing.Pool().map(): (1)
    map() only handles functions that take a single argument, and (2)
    this function can generally be imported from a module, as required
    by map().
    """
    print(f"all_args: {all_args}")
    (func1d, axis, arr, args, kwargs) = all_args
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)


scoring_functions = {"entropy_w_label_counts": entropy_w_label_counts,
                     "entropy_w_label_probs": entropy_w_label_probs}