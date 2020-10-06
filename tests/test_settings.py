import pytest

import inclusivekinematicfit


def test_set_setting():

    test_key = "test"
    test_value = [1, 2, 3]

    inclusivekinematicfit.settings.set_setting(test_key, test_value)
    assert (
        inclusivekinematicfit.settings._kinfit_global_settings[test_key] == test_value
    )


def test_get_setting():
    test_key = "test"
    test_value = [1, 2, 3]

    inclusivekinematicfit.settings._kinfit_global_settings[test_key] = test_value

    assert inclusivekinematicfit.settings.get_setting(test_key) == test_value


def test_set_setting_warning():
    test_key = "test"
    test_value = [1, 2, 3]

    inclusivekinematicfit.settings._kinfit_global_settings[test_key] = test_value
    with pytest.warns(inclusivekinematicfit.settings.OverwritingSettingsWarning):
        inclusivekinematicfit.settings.set_setting(test_key, test_value)


def test_get_setting_exception():
    with pytest.raises(ValueError):
        inclusivekinematicfit.settings.get_setting("not_set")
