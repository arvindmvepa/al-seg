from scipy.stats import entropy as entropy_func
import multiprocessing
import numpy as np


def mean_probs(im_labels):
    "Get the mean of the image label scores"
    # Expect an input shape of (256, 256) with each pixel representing the score in that pixel
    im_labels = np.concatenate(im_labels)
    # Take the average bald score across the image
    mean_probs = np.mean(im_labels.flatten())
    return mean_probs


def entropy_w_label_probs(im_labels):
    im_labels = np.concatenate(im_labels)
    # take the average probability for each class and calculate entropy from that
    im_labels = np.mean(im_labels, axis=0)
    # flatten image dimensions
    im_labels = im_labels.reshape((im_labels.shape[0], -1))

    entropy_arr = parallel_apply_along_axis(pixel_entropy_w_probs, 0, im_labels)
    mean_entropy = np.mean(entropy_arr)
    return mean_entropy


def entropy_w_label_counts(im_labels):
    im_labels = im_labels.reshape((im_labels.shape[0], -1))
    entropy_arr = parallel_apply_along_axis(pixel_entropy_w_label_counts, 0, im_labels)
    mean_entropy = np.mean(entropy_arr)
    return mean_entropy


def ensemble_variance_ratio(im_labels):
    n_members = len(im_labels)

    im_labels = np.concatenate(im_labels)

    # Convert probabilities to class indices
    im_class_labels = im_labels.argmax(axis=1)

    # Count the number of times each class is the modal class, pixel-wise
    # This gives an array of shape (n_members, 256, 256)
    fm = np.apply_along_axis(lambda x: np.bincount(x, minlength=n_members).max(), 
                             axis=0, 
                             arr=im_class_labels)

    # Compute variance ratio
    v = 1 - fm / n_members

    average_disperson = np.mean(v)
    return average_disperson


def pixel_entropy_w_probs(pixel_probs):
    return entropy_func(pixel_probs)

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

    # filter chunks if len(sub_arr) = 0
    chunks = [chunk for chunk in chunks if len(chunk[2]) > 0]

    pool = multiprocessing.Pool()
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
    (func1d, axis, arr, args, kwargs) = all_args
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)


scoring_functions = {"entropy_w_label_counts": entropy_w_label_counts,
                     "entropy_w_label_probs": entropy_w_label_probs,
                     "ensemble_variance_ratio": ensemble_variance_ratio,
                     "mean_probs": mean_probs}
