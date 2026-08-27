from scripts.run_phase3_devset_baselines import COMMANDS, command_label


def test_baseline_command_labels_and_unit_shape():
    assert [command_label(x) for x in COMMANDS] == ["00deg", "05deg", "10deg"]
    assert len(COMMANDS) == 3
