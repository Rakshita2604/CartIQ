from cartiq.query.inflection import variants

VOCAB = {"masala", "masalas", "egg", "eggs", "oils", "oil", "biscuits", "tomatoes", "dal", "gas"}


def test_adds_forms_present_in_the_vocabulary() -> None:
    assert variants("masala", VOCAB) == ["masalas"]
    assert variants("eggs", VOCAB) == ["egg"]
    assert variants("oil", VOCAB) == ["oils"]
    assert variants("tomato", VOCAB) == ["tomatoes"]


def test_nothing_when_no_form_exists() -> None:
    assert variants("milk", VOCAB) == []
    assert variants("biscuit", set()) == []


def test_short_or_non_alphabetic_terms_are_left_alone() -> None:
    assert variants("ga", VOCAB) == []
    assert variants("500ml", {"500mls"}) == []
