import numpy as np
import pytest
from birdsafe.fusion import Observation, fuse
from birdsafe.synth import make_pane

CANVAS_300_MM = (0.0, 0.0, 300.0, 300.0)


def _observation(pane, origin_mm=(0.0, 0.0)):
    return Observation(pane.markers, pane.unknown, origin_mm, pane.mm_per_px)


def test_one_aligned_observation_reproduces_its_masks():
    p = make_pane(300, 300, 40, 6, 0.5, unknown_boxes_mm=((100.0, 100.0, 60.0, 60.0),))
    fused = fuse([_observation(p)], CANVAS_300_MM)
    assert (fused.markers == (p.markers & ~p.unknown)).all()
    assert (fused.unknown == p.unknown).all()
    assert fused.frames == 1


def test_opencv_style_uint8_masks_give_the_same_answer_as_booleans():
    p = make_pane(300, 300, 40, 6, 0.5, unknown_boxes_mm=((100.0, 100.0, 60.0, 60.0),))
    as_u8 = Observation(p.markers.astype(np.uint8) * 255, p.unknown.astype(np.uint8) * 255,
                        (0.0, 0.0), p.mm_per_px)
    from_u8 = fuse([as_u8], CANVAS_300_MM)
    from_bool = fuse([_observation(p)], CANVAS_300_MM)
    assert from_u8.markers.dtype == bool and from_u8.unknown.dtype == bool
    assert (from_u8.markers == from_bool.markers).all()
    assert (from_u8.unknown == from_bool.unknown).all()


def test_two_glare_boxes_leave_only_their_overlap_unknown():
    a = make_pane(300, 300, 40, 6, 0.5, unknown_boxes_mm=((50.0, 50.0, 100.0, 100.0),))
    b = make_pane(300, 300, 40, 6, 0.5, unknown_boxes_mm=((100.0, 100.0, 100.0, 100.0),))
    one_frame = fuse([_observation(a)], CANVAS_300_MM)
    two_frames = fuse([_observation(a), _observation(b)], CANVAS_300_MM)
    assert (two_frames.unknown == (a.unknown & b.unknown)).all()
    assert two_frames.unknown.mean() < one_frame.unknown.mean()
    # A dot hidden by one frame's glare is kept when the other frame saw it.
    assert (two_frames.markers == ((a.markers & ~a.unknown) | (b.markers & ~b.unknown))).all()


def test_a_marker_seen_under_glare_does_not_hide_a_gap_another_frame_resolved_as_empty():
    glare = make_pane(300, 300, 40, 6, 0.5, omit=((3, 3),),
                      unknown_boxes_mm=((110.0, 110.0, 60.0, 60.0),))
    clear = make_pane(300, 300, 40, 6, 0.5, omit=((3, 3),))
    spurious = glare.markers.copy()
    spurious[258:262, 258:262] = True  # detection at the planted gap centre, 130 mm
    glare_frame = Observation(spurious, glare.unknown, (0.0, 0.0), 0.5)
    fused = fuse([glare_frame, _observation(clear)], CANVAS_300_MM)
    assert not fused.markers[258:262, 258:262].any()
    assert not fused.unknown[258:262, 258:262].any()


def test_an_offset_observation_lands_on_the_expected_canvas_pixels():
    markers = np.zeros((400, 400), bool)
    markers[10:14, 20:24] = True
    # Observation pixel (0, 0) is board (50, 25) mm, which is canvas pixel (100, 50) at 0.5 mm/px.
    obs = Observation(markers, np.zeros((400, 400), bool), (-50.0, -25.0), 0.5)
    fused = fuse([obs], CANVAS_300_MM)
    assert fused.markers.sum() == 16
    assert fused.markers[60:64, 120:124].all()
    assert not fused.unknown[50:450, 100:500].any()
    assert fused.unknown[:50].all() and fused.unknown[450:].all()
    assert fused.unknown[:, :100].all() and fused.unknown[:, 500:].all()


def test_a_canvas_inside_a_larger_observation_reads_the_matching_crop():
    p = make_pane(300, 300, 40, 6, 0.5, unknown_boxes_mm=((110.0, 60.0, 30.0, 30.0),))
    fused = fuse([_observation(p)], (100.0, 50.0, 100.0, 100.0))
    assert (fused.markers == (p.markers & ~p.unknown)[100:300, 200:400]).all()
    assert (fused.unknown == p.unknown[100:300, 200:400]).all()
    assert fused.unknown.any()


def test_an_observation_wholly_outside_the_canvas_sees_nothing():
    p = make_pane(100, 100, 40, 6, 0.5)
    fused = fuse([_observation(p)], (500.0, 500.0, 100.0, 100.0))
    assert fused.unknown.all() and not fused.markers.any()
    assert fused.frames == 1


def test_no_observations_gives_an_all_unknown_canvas():
    fused = fuse([], (0.0, 0.0, 100.0, 50.0), mm_per_px=0.5)
    assert fused.unknown.shape == (100, 200)
    assert fused.unknown.all() and not fused.markers.any()
    assert fused.frames == 0


def test_no_observations_and_no_scale_is_an_error():
    with pytest.raises(ValueError, match="mm_per_px"):
        fuse([], CANVAS_300_MM)


def test_mixed_scales_raise():
    fine = make_pane(100, 100, 40, 6, 0.25)
    coarse = make_pane(100, 100, 40, 6, 0.5)
    with pytest.raises(ValueError, match="mm_per_px"):
        fuse([_observation(fine), _observation(coarse)], CANVAS_300_MM)
