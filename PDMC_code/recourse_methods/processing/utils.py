from typing import Optional, Dict


def merge_default_parameters(hyperparams: Optional[Dict], default: Dict) -> Dict:
    """
    Checks if the input parameter hyperparams contains every necessary key and if not, uses default values or
    raises a ValueError if no default value is given.

    Parameters
    ----------
    hyperparams: dict
        Hyperparameter as passed to the recourse method.
    default: dict
        Dictionary with every necessary key and default value.
        If key has no default value and hyperparams has no value for it, raise a ValueError

    Returns
    -------
    dict
        Dictionary with every necessary key.
    """
    if hyperparams is None:
        return default

    keys = default.keys()
    dict_output = dict()

    for key in keys:
        if isinstance(default[key], dict):
            hyperparams[key] = (
                dict() if key not in hyperparams.keys() else hyperparams[key]
            )
            sub_dict = merge_default_parameters(hyperparams[key], default[key])
            dict_output[key] = sub_dict
            continue
        if key not in hyperparams.keys():
            default_val = default[key]
            if default_val is None:
                # None value for key depicts that user has to pass this value in hyperparams
                raise ValueError(
                    "For {} is no default value defined, please pass this key and its value in hyperparams".format(
                        key
                    )
                )
            elif isinstance(default_val, str) and default_val == "_optional_":
                # _optional_ depicts that value for this key is optional and therefore None
                default_val = None
            dict_output[key] = default_val
        else:
            if hyperparams[key] is None:
                raise ValueError("For {} in hyperparams is a value needed".format(key))
            dict_output[key] = hyperparams[key]

    return dict_output
