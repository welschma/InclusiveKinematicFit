import pytest

import kinfit


def test_set_setting():

    test_key = "test"
    test_value = [1, 2, 3]

    kinfit.settings.set_setting(test_key, test_value)
    assert kinfit.settings._kinfit_global_settings[test_key] == test_value


def test_get_setting():
    test_key = "test"
    test_value = [1, 2, 3]

    kinfit.settings._kinfit_global_settings[test_key] = test_value

    assert kinfit.settings.get_setting(test_key) == test_value


def test_set_setting_warning():
    test_key = "test"
    test_value = [1, 2, 3]

    kinfit.settings._kinfit_global_settings[test_key] = test_value
    with pytest.warns(kinfit.settings.OverwritingSettingsWarning):
        kinfit.settings.set_setting(test_key, test_value)


def test_get_setting_exception():
    with pytest.raises(ValueError):
        kinfit.settings.get_setting("not_set")
