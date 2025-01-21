def round_dict_values(dictionary, n=3):
    """Round dictionary to `n` decimal places."""
    # TODO check if complex number can be converted to float.
    # for val in dictionary.values():
    # TODO guard for numbers, but needs to allow an empty list
    # TODO maybe also need None ^^
    #     if not isinstance(val, (int, float, complex)):
    #         print(dictionary)
    #         raise ValueError("{0} is not numeric".format(val))

    # rounded = {key: float(f"{val:.{n}f}") for key, val in dictionary.items()}
    rounded = {
        key: round(value, 2) if isinstance(value, (int, float)) else value
        for key, value in dictionary.items()
    }
    return rounded


def is_even(num: int) -> bool:
    return True if num == 0 else num % 2 == 0
