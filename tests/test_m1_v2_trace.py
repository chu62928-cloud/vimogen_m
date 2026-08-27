import torch

from sampling.m1_guidance import M1Config, M1TraceRecorder


def _payload():
    return {
        "sigma": 0.5,
        "timestep": 12.0,
        "x_sigma": torch.ones(1, 3, 276),
        "v_cfg": torch.full((1, 3, 276), 2.0),
        "x0_hat": torch.full((1, 3, 276), 3.0),
        "x0_guided": torch.full((1, 3, 276), 4.0),
        "x0_reconciled": torch.full((1, 3, 276), 5.0),
        "v_corrected": torch.full((1, 3, 276), 6.0),
        "x_next": torch.full((1, 3, 276), 7.0),
        "next_model_x0": torch.full((1, 3, 276), 8.0),
    }


def test_trace_is_opt_in_and_disabled_recorder_stays_empty():
    assert M1Config.from_mapping({}).trace_enabled is False
    recorder = M1TraceRecorder(enabled=False)
    recorder.record_step(**_payload())
    assert recorder.records == []


def test_trace_recorder_captures_required_cpu_snapshots_and_algebraic_fields():
    recorder = M1TraceRecorder(enabled=True)
    recorder.record_step(**_payload())
    assert len(recorder.records) == 1
    record = recorder.records[0]
    assert set(record) >= {
        "sigma", "timestep", "x_sigma", "v_cfg", "x0_hat", "x0_guided",
        "x0_reconciled", "v_corrected", "x_next", "next_model_x0",
    }
    assert record["x_sigma"].device.type == "cpu"
    assert torch.equal(record["x0_reconciled"], torch.full((1, 3, 276), 5.0))
