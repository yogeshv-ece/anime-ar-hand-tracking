"""
Sanity tests for the non-webcam-dependent logic, using synthetic
21-landmark hands so we can verify geometry/gesture/state-machine
correctness without a physical camera.
"""
import numpy as np

from core.hand_geometry import (
    HandFrame, Handedness, compute_palm_center, compute_hand_orientation,
    WRIST, THUMB_TIP, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP,
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP,
    RING_MCP, RING_PIP, RING_DIP, RING_TIP,
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP, THUMB_MCP,
)
from core.gestures import recognize_chidori, recognize_rasengan_shape, compute_two_hand_geometry
from core.state_machine import GestureStateMachine, AppState, RasenganRotationTracker
from core.filters import OneEuroFilter, AngleEMA


def make_hand(finger_extended: dict, wrist=(320, 400), handedness=Handedness.RIGHT):
    """
    Build a synthetic, upright hand (pointing straight up in image space)
    with each finger either extended (straight up) or folded (bent back
    toward the wrist), based on `finger_extended` dict keys:
    thumb/index/middle/ring/pinky -> bool.
    """
    lm = np.zeros((21, 2), dtype=np.float32)
    wx, wy = wrist
    lm[WRIST] = [wx, wy]

    def set_finger(mcp, pip, dip, tip, x_offset, extended):
        mcp_y = wy - 60
        lm[mcp] = [wx + x_offset, mcp_y]
        if extended:
            lm[pip] = [wx + x_offset, mcp_y - 30]
            lm[dip] = [wx + x_offset, mcp_y - 55]
            lm[tip] = [wx + x_offset, mcp_y - 80]
        else:
            # folded back toward wrist/palm
            lm[pip] = [wx + x_offset, mcp_y + 15]
            lm[dip] = [wx + x_offset * 0.8, mcp_y + 25]
            lm[tip] = [wx + x_offset * 0.6, mcp_y + 20]

    set_finger(INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, -30, finger_extended.get("index", False))
    set_finger(MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, 0, finger_extended.get("middle", False))
    set_finger(RING_MCP, RING_PIP, RING_DIP, RING_TIP, 10, finger_extended.get("ring", False))
    set_finger(PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP, 30, finger_extended.get("pinky", False))

    # thumb: sideways extension test
    lm[THUMB_MCP] = [wx - 45, wy - 20]
    if finger_extended.get("thumb", False):
        lm[THUMB_TIP] = [wx - 90, wy - 30]  # far sideways
    else:
        lm[1] = [wx - 45, wy - 20]  # THUMB_CMC
        lm[THUMB_TIP] = [wx - 40, wy - 10]  # tucked close to palm

    z = np.zeros(21, dtype=np.float32)
    return HandFrame(landmarks_px=lm, landmarks_z=z, handedness=handedness, detection_confidence=0.95)


def test_palm_center_and_orientation():
    hand = make_hand({"index": True, "middle": False, "ring": False, "pinky": False})
    center = hand.palm_center
    assert center.shape == (2,)
    # Upright hand -> orientation should be close to 0 degrees
    assert abs(hand.orientation_deg) < 5, f"expected ~0 deg, got {hand.orientation_deg}"
    print(f"[OK] palm_center={center}, orientation={hand.orientation_deg:.2f} deg")


def test_finger_states():
    hand = make_hand({"index": True, "middle": False, "ring": False, "pinky": False, "thumb": False})
    f = hand.fingers
    print(f"[fingers] {f}")
    assert f.index is True, "index should be detected as extended"
    assert f.middle is False, "middle should be detected as folded"
    assert f.ring is False, "ring should be detected as folded"
    assert f.pinky is False, "pinky should be detected as folded"
    print("[OK] finger state detection matches expected pattern")


def test_chidori_recognition():
    chidori_hand = make_hand({"index": True, "middle": False, "ring": False, "pinky": False})
    assert recognize_chidori(chidori_hand) is True, "should recognize chidori pattern"

    open_hand = make_hand({"index": True, "middle": True, "ring": True, "pinky": True})
    assert recognize_chidori(open_hand) is False, "open hand should NOT match chidori"
    print("[OK] chidori gesture recognition correct (positive + negative case)")


def test_rasengan_recognition():
    left = make_hand({"index": True, "middle": True, "ring": True, "pinky": True},
                      wrist=(250, 400), handedness=Handedness.LEFT)
    right = make_hand({"index": True, "middle": True, "ring": True, "pinky": True},
                       wrist=(390, 400), handedness=Handedness.RIGHT)
    assert recognize_rasengan_shape(left, right) is True, "two open hands at plausible distance should match"

    geom = compute_two_hand_geometry(left, right)
    expected_mid = (left.palm_center + right.palm_center) / 2.0
    assert np.allclose(geom.midpoint, expected_mid)
    print(f"[OK] rasengan shape recognized, midpoint={geom.midpoint}, dist={geom.distance:.1f}")

    # too far apart -> should NOT match
    far_right = make_hand({"index": True, "middle": True, "ring": True, "pinky": True},
                           wrist=(1200, 400), handedness=Handedness.RIGHT)
    assert recognize_rasengan_shape(left, far_right) is False, "hands too far apart should not match"
    print("[OK] rasengan correctly rejects hands that are too far apart")


def test_state_machine_debounce():
    sm = GestureStateMachine()
    # feed chidori_raw=True for fewer frames than required -> should stay IDLE
    for _ in range(3):
        state = sm.update(chidori_raw=True, rasengan_raw=False, any_hand_tracked=True)
    assert state == AppState.IDLE, "should not activate before debounce threshold"

    for _ in range(5):
        state = sm.update(chidori_raw=True, rasengan_raw=False, any_hand_tracked=True)
    assert state == AppState.CHIDORI_ACTIVE, f"should be active after enough frames, got {state}"
    print("[OK] chidori debounce activates only after required consecutive frames")

    # a single dropped frame shouldn't deactivate (flicker resistance)
    state = sm.update(chidori_raw=False, rasengan_raw=False, any_hand_tracked=True)
    assert state == AppState.CHIDORI_ACTIVE, "single dropped frame should not deactivate"
    print("[OK] single-frame flicker does not deactivate the gesture")

    for _ in range(10):
        state = sm.update(chidori_raw=False, rasengan_raw=False, any_hand_tracked=True)
    assert state == AppState.IDLE, "should deactivate after sustained absence"
    print("[OK] gesture deactivates after sustained absence")


def test_rasengan_rotation_energy():
    tracker = RasenganRotationTracker()
    t = 0.0
    # simulate steadily increasing angle -> rotation -> energy should rise
    angle = 0.0
    for i in range(30):
        t += 1 / 30.0
        angle += 20  # 20 deg per frame -> fast rotation
        tracker.update(angle, now=t)
    assert tracker.energy > 50, f"energy should have risen with sustained rotation, got {tracker.energy}"
    print(f"[OK] energy rose to {tracker.energy:.1f} under sustained rotation")

    # now hold still -> energy should decay
    for i in range(30):
        t += 1 / 30.0
        tracker.update(angle, now=t)  # angle unchanged
    assert tracker.energy < 20, f"energy should decay when hands stop rotating, got {tracker.energy}"
    print(f"[OK] energy decayed to {tracker.energy:.2f} after hands stopped")


def test_one_euro_filter_smooths_noise():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    t = 0.0
    outputs = []
    rng = np.random.default_rng(42)
    true_value = 100.0
    for i in range(60):
        t += 1 / 30.0
        noisy = true_value + rng.normal(0, 5)
        outputs.append(f.filter(noisy, timestamp=t))
    # variance of filtered output should be substantially lower than raw noise variance
    raw_var = 25.0  # sigma=5 -> var=25
    filtered_var = np.var(outputs[10:])  # skip warmup
    assert filtered_var < raw_var, f"filter should reduce variance, got {filtered_var}"
    print(f"[OK] OneEuroFilter reduced noise variance to {filtered_var:.2f} (raw ~{raw_var})")


def test_angle_ema_wraparound():
    ema = AngleEMA(alpha=0.5)
    ema.filter(179.0)
    result = ema.filter(-179.0)  # only a 2-degree true jump, not 358
    assert abs(result - 180) < 10 or abs(result + 180) < 10, f"should handle wraparound, got {result}"
    print(f"[OK] AngleEMA handled wraparound correctly, result={result:.2f}")


if __name__ == "__main__":
    tests = [
        test_palm_center_and_orientation,
        test_finger_states,
        test_chidori_recognition,
        test_rasengan_recognition,
        test_state_machine_debounce,
        test_rasengan_rotation_energy,
        test_one_euro_filter_smooths_noise,
        test_angle_ema_wraparound,
    ]
    for t in tests:
        print(f"\n--- {t.__name__} ---")
        t()
    print("\nALL TESTS PASSED")
