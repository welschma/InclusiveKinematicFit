"""This module manages the handling of global settings for the kinematic fit.
"""
import warnings

from typing import Any, Dict

# the global object hosting the current settings
_kinfit_global_settings = {}  # type: Dict[str, Any]


def get_setting(key: str) -> Any:
    """Returns the value of the setting specified by `key`.

    :param key: Setting Name
    :type key: str
    :raises ValueError: Raised if specified setting doesn't exist.
    :return: Setting Value
    :rtype: Any
    """
    try:
        return _kinfit_global_settings[key]
    except KeyError as exc:
        raise ValueError(f"No settings found for setting {key}!") from exc


def set_setting(key: str, value: Any) -> None:
    """Set the setting with the specified name.
    """

    if key in _kinfit_global_settings.keys():
        warnings.warn(
            f"You're overwriting the following setting {key}: {_kinfit_global_settings[key]} -> {value}.",
            OverwritingSettingsWarning,
        )

    _kinfit_global_settings[key] = value


class OverwritingSettingsWarning(RuntimeWarning):
    pass
