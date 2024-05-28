import numpy as np

def check_list(vars=None, var=None):
    """ Type-checks passed variables and wraps each non-list in a list.
        Specify either vars as tuple to pass multiple variables, or var to pass
        a single variable (ensures correct behavior if var is tuple). Nones are
        always returned as Nones. Enables functions that expect iterable
        arguments (e.g. in single for-loops) to handle scalar inputs.

    Parameters
    ----------
    vars : tuple of arbitrary types (m,)
        Multiple variables, each of which is type-checked and wrapped if
        required. Elements of vars that are tuples are handled as expected.
        Takes precedence over var if both are specified. The default is None.
    var : arbitrary type
        Single variable to be type-checked and wrapped if required. Serves to
        distinguish between a tuple of multiple variables and a single variable
        of tuple type. The default is None.

    Returns
    -------
    vars : tuple of lists or Nones (m,)
        Multiple type-checked and wrapped variables, ready for unpacking.
    var : list or None
        Type-checked and wrapped variable.
    """
    if vars is not None:
        return (v if v is None or isinstance(v, list) else [v] for v in vars)
    elif var is not None:
        return var if var is None or isinstance(var, list) else [var]


def format_sequence(sequence, delimit=', ', final=None):
    """ Returns a single formatted string listing the elements of a sequence.
        Sequence elements are separated by a given delimiter, and the final
        element can be preceded by an additional conjunction to close the list.

    Parameters
    ----------
    sequence : tuple or list or array or set of arbitrary shape and types
        Collection of elements to be listed in string format.
    delimit : str, optional
        Delimiter text to separate sequence elements. Delimiter is not modified
        before joining the elements, so spaces should be included if desired.
        The default is ', '. 
    final : str, optional
        If specified, a conjunction to insert before the final element of the
        sequence (but after the preceding delimiter). Conjunction is expanded
        by a single trailing space if it does not already end with one. Can be
        used to close listings of several elements by "and" or "or", or to add
        "at least" or "at most" before a single element. The default is None.

    Returns
    -------
    str
        Formatted string listing all elements of the sequence.
    """    
    strings = [str(entry) for entry in sequence]
    if final is not None:
        final += '' if final.endswith(' ') else ' '
        strings[-1] = final + strings[-1]
    return delimit.join(strings)


def ensure_array(vars=None, var=None, copy=False, dtype=float,
                 dims=None, shape=None, list_shapes=False, verbose=False):
    """ Turns variables to arrays, validates dimensionality, and adjusts shape.
        Calls np.array() with the specified dtype on the given variables. Can
        be set to break if one of the resulting arrays has an unexpected number
        of dimensions. Validated arrays can then be rearranged into a given
        shape. Specify either vars as tuple to pass multiple variables, or var
        to pass a single variable (ensures correct behavior if var is tuple).

    Parameters
    ----------
    vars : tuple or list of arbitrary types (m,), optional
        Multiple variables to be type-checked. Elements of vars that are tuples
        are handled as expected. The default is None.
    var : arbitrary type, optional
        Single variable to be type-checked. Serves to distinguish between a
        tuple of multiple variables and a single variable of tuple type. The
        default is None.
    copy : bool, optional
        If True, calls np.array() on every variable, including those that are
        already arrays. If False, calls np.array() only on non-array variables.
        The default is False.
    dtype : str, optional
        If specified, calls np.array(dtype=dtype) where requested, depending on
        the copy parameter. Else, set by np.array() to fit the contents of each
        variable it is called upon. Forcing a given dtype may cause loss of
        precision. The default is float.
    dims : int or tuple or list or 1D array of ints (n,), optional
        If specified, ensures that the number of dimensions of each (passed or
        newly created) array is one of the specified values. Raises an error
        for arrays with unexpected dimensionality. The default is None.
    shape : tuple or list or 1D array of ints (p,), optional
        If specified, attempts to force each array into the desired shape and
        fails with an error if broadcast is not possible. By np.reshape()
        default, one dimension can be set to -1 to be inferred from remaining
        dimensions. In addition, one or several dimensions can be unspecified
        as None. If an array has the same number of dimensions as the requested
        shape, unspecified dimensions are adopted from the initial shape of the
        array. If an array has as many dimensions as the requested shape minus
        the number of unspecified dimensions, each unspecified dimension is set
        to 1. All other cases are considered ambiguous and raise an error.
        Unspecified dimensions can only be used to expand from a single lower
        dimension (e.g. 1D) to some higher dimension (e.g. 2D), but not from
        e.g. 1D to 3D and 2D to 3D at the same time. Use (-1, None) to turn 1D
        arrays into single-column 2D arrays (2D arrays are accepted normally as
        they are). Use (None, -1) to turn 1D arrays into single-row 2D arrays.
        The default is None.
    list_shapes : bool, optional
        If True, returns the initial shape of each array (after conversion and
        dimensionality validation, but before reshaping) in addition to the
        arrays themselves. Can be used to retrieve the original shape of
        converted non-array variables, if required. The default is False.
    verbose : bool, optional
        If True, prints a warning every time the shape of an array has been
        changed, logging the initial and the new shape. The default is False.

    Returns
    -------
    arrays : array or list (m,) of arrays (arbitrary shape)
        One or several variables in array format with validated dimensionality.
        Single variables are returned as an unwrapped array. Multiple variables
        are returned as a list of arrays, ready for unpacking.
    input_shapes : tuple or list (m,) of tuples (arbitrary length)
        Original shape of each variable in array format (after conversion, but
        before reshaping). Only returned if list_shapes is True.

    Raises
    ------
    ValueError
        Breaks if an array has invalid dimensionality (before reshaping).
    ValueError
        Breaks if unspecified dimensions (None) in the requested shape are
        ambiguous for the initial shape of the array. Happens when the desired
        number of dimensions is not equal to the initial number of dimensions,
        or to the initial number of dimensions minus the number of unspecified
        dimensions.
    """    
    # Input interpretation:
    if var is not None:
        vars = (var,)
    # Prepare dimensionality validation:
    if dims is not None:
        # Assert iterable:
        if not isinstance(dims, (list, tuple)):
            dims = (dims,)
        # List accepted dimensionality to be printed with error message:
        valid = format_sequence(dims, delimit=', ' if len(dims) > 2 else ' ',
                                final=None if len(dims) == 1 else 'or')
        msg_dims = f"Number of array dimensions must be {valid}."
    # Prepare reshaping:
    if shape is not None:
        shape = np.array(shape)
        buffer_dims = shape == None
        n_buffers = sum(buffer_dims)

    # Run type-checks:
    arrays, input_shapes = [], []
    for var in vars:
        # Force into array format:
        if copy or not isinstance(var, np.ndarray):
            var = np.array(var, dtype)
        # Optional dimensionality validation:
        if dims is not None and var.ndim not in dims:
            raise ValueError(msg_dims)
        # Log initial array shape:
        input_shapes.append(var.shape)
        # Optional reshaping:
        if shape is not None:
            initial = np.array(var.shape)
            target = shape.copy()
            # Manage unspecified dimensions:    
            if n_buffers and initial.size == shape.size:
                # Preserve shapes of matching dimensionality:
                target[buffer_dims] = initial[buffer_dims]
            elif n_buffers and initial.size == shape.size - n_buffers:
                # Expand if possible:
                target[buffer_dims] = 1
            elif n_buffers:
                # Report failure to infer unspecified dimensions:
                msg_shape = f'Unspecified dimensions in {target} are '\
                            f'ambiguous for initial shape {initial}.' 
                raise ValueError(msg_shape)
            var = var.reshape(target)
            if verbose and not np.array_equal(initial, var.shape):
                # Report deviation from initial array shape:
                print(f'WARNING: Reshaped {tuple(initial)} to {var.shape}.')
            if len(vars) == 1:
                # Single variable early exit:
                return (var, input_shapes[0]) if list_shapes else var
        # Log finalized array:
        arrays.append(var)
    return (*arrays, input_shapes) if list_shapes else arrays

