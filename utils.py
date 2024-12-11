def round_dict_values(dictionary, n=3):
    """Round dictionary to `n` decimal places."""
    # TODO check if complex number can be converted to float.
    for val in dictionary.values():
        if not isinstance(val, (int, float, complex)):
            raise ValueError('{0} is not numeric'.format(val))
    
    rounded = {key: float(f"{val:.{n}f}") for key, val in dictionary.items()}
    return rounded