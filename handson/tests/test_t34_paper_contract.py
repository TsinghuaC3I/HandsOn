"""Known regressions between the collected/paper task and the packaged task."""
import numpy as np

from handson.sim.isaaclab import ycb01_semantics as gate


def test_center_past_box_face_is_not_a_complete_ball_pass():
    frame = np.array([0., 0., 0., 1., 0., 0., 0.])
    partial = gate.ycb01_crossing_metrics(
        np.array([0., -.05, .033]), np.array([0., .04, .033]), frame)
    assert partial['center_crossed_now']
    assert not partial['full_passed_now']
    complete = gate.ycb01_crossing_metrics(
        np.array([0., .04, .033]), np.array([0., .08, .033]), frame)
    assert complete['full_pass_crossed_now']
    assert complete['full_pass_inside_opening_now']


def test_fast_diagonal_escape_does_not_clear_the_opening():
    frame = np.array([0., 0., 0., 1., 0., 0., 0.])
    metrics = gate.ycb01_crossing_metrics(
        np.array([0., -.02, .033]), np.array([.15, .10, .033]), frame)
    assert metrics['full_pass_crossed_now']
    assert not metrics['full_pass_inside_opening_now']


def test_historical_reset_uses_start_side_and_randomizes_gate():
    frames = []
    for seed in range(100):
        layout = gate.sample_ycb01_layout(np.random.RandomState(seed),
                                         randomize=True, table_top_z=.89)
        local = gate.world_to_ycb01_gate(layout[gate.YCB01_BALL_NAME][:3],
                                       layout['gate_frame'])
        assert -.30 <= local[1] <= -.08
        assert abs(local[0]) <= .08
        frames.append(layout['gate_frame'])
    assert np.ptp(np.array(frames)[:, 0]) > .005
    assert np.ptp(np.array(frames)[:, 1]) > .02
    assert np.ptp(np.array(frames)[:, 6]) > .05
