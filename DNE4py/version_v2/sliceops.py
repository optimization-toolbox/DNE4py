import numpy as np


def multi_slice_add(x1_inplace, x2, x1_slices=(), x2_slices=()):
    """
    Does an inplace addition on x1 given a list of slice objects
    If slices for both are given, it is assumed that they will be of
    the same size and each slice will have the same # of elements
    """

    if (len(x1_slices) != 0) and (len(x2_slices) == 0):
        for x1_slice in x1_slices:
            x1_inplace[x1_slice] += x2

    elif (len(x1_slices) != 0) and (len(x2_slices) != 0) \
            and (len(x2_slices) == len(x1_slices)):

        for i in range(len(x1_slices)):
            x1_inplace[x1_slices[i]] += x2[x2_slices[i]]

    elif (len(x1_slices) == 0) and (len(x2_slices) == 0):
        x1_inplace += x2


def multi_slice_subtract(x1_inplace, x2, x1_slices=(), x2_slices=()):
    """
    Does an inplace addition on x1 given a list of slice objects
    If slices for both are given, it is assumed that they will be of
    the same size and each slice will have the same # of elements
    """

    if (len(x1_slices) != 0) and (len(x2_slices) == 0):
        for x1_slice in x1_slices:
            x1_inplace[x1_slice] -= x2

    elif (len(x1_slices) != 0) and (len(x2_slices) != 0) \
            and (len(x2_slices) == len(x1_slices)):

        for i in range(len(x1_slices)):
            x1_inplace[x1_slices[i]] -= x2[x2_slices[i]]

    elif (len(x1_slices) == 0) and (len(x2_slices) == 0):
        x1_inplace -= x2


def multi_slice_multiply(x1_inplace, x2, x1_slices=(), x2_slices=()):
    """
    Does an inplace multiplication on x1 given a list of slice objects
    If slices for both are given, it is assumed that they will be of
    the same size and each slice will have the same # of elements
    """

    if (len(x1_slices) != 0) and (len(x2_slices) == 0):
        for x1_slice in x1_slices:
            x1_inplace[x1_slice] *= x2

    elif (len(x1_slices) != 0) and (len(x2_slices) != 0) \
            and (len(x2_slices) == len(x1_slices)):

        for i in range(len(x1_slices)):
            x1_inplace[x1_slices[i]] *= x2[x2_slices[i]]

    elif (len(x1_slices) == 0) and (len(x2_slices) == 0):
        x1_inplace *= x2


def multi_slice_divide(x1_inplace, x2, x1_slices=(), x2_slices=()):
    """
    Does an inplace multiplication on x1 given a list of slice objects
    If slices for both are given, it is assumed that they will be of
    the same size and each slice will have the same # of elements
    """

    if (len(x1_slices) != 0) and (len(x2_slices) == 0):
        for x1_slice in x1_slices:
            x1_inplace[x1_slice] /= x2

    elif (len(x1_slices) != 0) and (len(x2_slices) != 0) \
            and (len(x2_slices) == len(x1_slices)):

        for i in range(len(x1_slices)):
            x1_inplace[x1_slices[i]] /= x2[x2_slices[i]]

    elif (len(x1_slices) == 0) and (len(x2_slices) == 0):
        x1_inplace /= x2


def multi_slice_assign(x1_inplace, x2, x1_slices=(), x2_slices=()):
    """
    Does an inplace assignment on x1 given a list of slice objects
    If slices for both are given, it is assumed that they will be of
    the same size and each slice will have the same # of elements
    """

    if (len(x1_slices) != 0) and (len(x2_slices) == 0):
        for x1_slice in x1_slices:
            x1_inplace[x1_slice] = x2

    elif (len(x1_slices) != 0) and (len(x2_slices) != 0) \
            and (len(x2_slices) == len(x1_slices)):

        for i in range(len(x1_slices)):
            x1_inplace[x1_slices[i]] = x2[x2_slices[i]]

    elif (len(x1_slices) == 0) and (len(x2_slices) == 0):
        x1_inplace = x2


def multi_slice_mod(x1_inplace, x2, x1_slices=(), x2_slices=()):
    """
    Does an inplace modulo on x1 given a list of slice objects
    If slices for both are given, it is assumed that they will be of
    the same size and each slice will have the same # of elements
    """

    if (len(x1_slices) != 0) and (len(x2_slices) == 0):
        for x1_slice in x1_slices:
            x1_inplace[x1_slice] %= x2

    elif (len(x1_slices) != 0) and (len(x2_slices) != 0) \
            and (len(x2_slices) == len(x1_slices)):

        for i in range(len(x1_slices)):
            x1_inplace[x1_slices[i]] %= x2[x2_slices[i]]

    elif (len(x1_slices) == 0) and (len(x2_slices) == 0):
        x1_inplace %= x2


def multi_slice_fabs(x1_inplace, x1_slices=()):
    """
    Does an inplace fabs on x1 given a list of slice objects
    """

    if len(x1_slices) != 0:
        for x1_slice in x1_slices:
            np.fabs(x1_inplace[x1_slice], out=x1_inplace[x1_slice])

    else:
        np.fabs(x1_inplace, out=x1_inplace)


def multi_slice_clip(x1_inplace, lower, upper, xslices=None,
                     lslices=None, uslices=None):
    """
    Does an inplace clip on x1
    """

    if (lslices is None) and (uslices is None) and (xslices is None):
        np.clip(x1_inplace, lower, upper, out=x1_inplace)

    elif (lslices is None) or (uslices is None) and (xslices is not None):
        for xslice in xslices:
            np.clip(x1_inplace[xslice], lower, upper, out=x1_inplace[xslice])

    elif (lslices is not None) and (uslices is not None) \
            and (len(lslices) == len(uslices) and (xslices is not None)):
        for i in range(len(xslices)):
            np.clip(x1_inplace[xslices[i]], lower[lslices[i]], upper[uslices[i]],
                    out=x1_inplace[xslices[i]])

    else:
        raise NotImplementedError("Invalid arguments in multi_slice_clip")


def random_slices(rng, iterator_size, size_of_slice, max_step=1):
    """
    Returns a list of slice objects given the size of the iterator it
    will be used for and the number of elements desired for the slice
    This will return additional slice each time it wraps around the
    iterator

    iterator_size - the number of elements in the iterator
    size_of_slice - the number of elements the slices will cover
    max_step - the maximum number of steps a slice will take.
                This affects the number of slice objects created, as
                larger max_step will create more wraps around the iterator
                and so return more slice objects

    The number of elements is not guaranteed when slices overlap themselves
    """

    step_size = rng.randint(1, max_step + 1)  # randint is exclusive
    start_step = rng.randint(0, iterator_size)

    return build_slices(start_step, iterator_size, size_of_slice, step_size)


def build_slices(start_step, iterator_size, size_of_slice, step_size):
    """
    Given a starting index, the size of the total members of the window,
    a step size, and the size of the iterator the slice will act upon,
    this function returns a list of slice objects that will cover that full
    window. Upon reaching the endpoints of the iterator, it will wrap around.
    """

    if step_size >= iterator_size:
        raise NotImplementedError("Error: step size must be less than the " +
                                  "size of the iterator")
    end_step = start_step + step_size * size_of_slice
    slices = []
    slice_start = start_step
    for i in range(1 + (end_step - step_size) // iterator_size):
        remaining = end_step - i * iterator_size
        if remaining > iterator_size:
            remaining = iterator_size

        slice_end = (slice_start + 1) + ((remaining -
                                          (slice_start + 1)) // step_size) * step_size
        slices.append(np.s_[slice_start:slice_end:step_size])
        slice_start = (slice_end - 1 + step_size) % iterator_size

    return slices


def match_slices(slice_list1, slice_list2):
    """
    Will attempt to create additional slices to match the # elements of
    each slice from list1 to the corresponding slice of list 2.
    Will fail if the total # elements is different for each list
    """

    slice_list1 = list(slice_list1)
    slice_list2 = list(slice_list2)
    if slice_size(slice_list1) == slice_size(slice_list2):
        slice_list1.reverse()
        slice_list2.reverse()
        new_list1_slices = []
        new_list2_slices = []

        while len(slice_list1) != 0 and len(slice_list2) != 0:
            slice_1 = slice_list1.pop()
            slice_2 = slice_list2.pop()
            size_1 = slice_size(slice_1)
            size_2 = slice_size(slice_2)

            if size_1 < size_2:
                new_slice_2, slice_2 = splice_slice(slice_2, size_1)
                slice_list2.append(slice_2)
                new_list2_slices.append(new_slice_2)
                new_list1_slices.append(slice_1)

            elif size_2 < size_1:
                new_slice_1, slice_1 = splice_slice(slice_1, size_2)
                slice_list1.append(slice_1)
                new_list1_slices.append(new_slice_1)
                new_list2_slices.append(slice_2)

            elif size_1 == size_2:
                new_list1_slices.append(slice_1)
                new_list2_slices.append(slice_2)

    else:
        raise AssertionError("Error: slices not compatible")

    return new_list1_slices, new_list2_slices


def splice_slice(slice_obj, num_elements):
    """
    Returns two slices spliced from a single slice.
    The size of the first slice will be # elements
    The size of the second slice will be the remainder
    """

    splice_point = slice_obj.step * (num_elements - 1) + slice_obj.start + 1
    new_start = splice_point - 1 + slice_obj.step
    return np.s_[slice_obj.start: splice_point: slice_obj.step], \
        np.s_[new_start: slice_obj.stop: slice_obj.step]


def slice_size(slice_objects):
    """
    Returns the total number of elements in the combined slices
    Also works if given a single slice
    """

    num_elements = 0

    try:
        for sl in slice_objects:
            num_elements += (sl.stop - (sl.start + 1)) // sl.step + 1
    except TypeError:
        num_elements += (slice_objects.stop - (slice_objects.start + 1)) \
            // slice_objects.step + 1

    return num_elements
