from nudemo.studies.batched_ingest import build_random_indices


def test_build_random_indices_spreads_samples_evenly() -> None:
    indices = build_random_indices(100, 5)

    assert indices == [0, 20, 40, 60, 99]


def test_build_random_indices_handles_small_sample_sets() -> None:
    indices = build_random_indices(3, 10)

    assert indices == [0, 1, 2]
