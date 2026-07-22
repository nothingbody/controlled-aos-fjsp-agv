from data.loader import _derived_extension_seed


def test_extension_seed_is_stable_and_instance_specific():
    first = _derived_extension_seed("data/benchmarks/brandimarte", "Mk01.fjs", 42)
    assert first == _derived_extension_seed(
        "data/benchmarks/brandimarte", "Mk01.fjs", 42
    )
    assert first != _derived_extension_seed(
        "data/benchmarks/brandimarte", "Mk02.fjs", 42
    )
    assert first != _derived_extension_seed(
        "data/benchmarks/hurink_edata", "Mk01.fjs", 42
    )
