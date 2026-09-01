# Anime AR — Chidori / Rasengan Hand-Tracked VFX Overlay

Real-time webcam gesture recognition + AR compositor. This project is
**only** the computer-vision system: hand detection, gesture recognition,
tracking, and VFX overlay. It does not generate the Chidori/Rasengan visuals
themselves — those are your After Effects exports, dropped into `assets/`.

## Setup

```bash
pip install -r requirements.txt

# One-time model download (required by the mediapipe Tasks API):
curl -L -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

Then run:

```bash
python main.py
```

Press `q` to quit, `d` to toggle the debug landmark/skeleton overlay.

## Why the Tasks API, not `mp.solutions.hands`

Older MediaPipe tutorials use `mp.solutions.hands.Hands(...)`. As of
mediapipe 0.10.x, `mp.solutions` is no longer part of the pip package —
importing it raises `AttributeError`. This project uses the current
`mediapipe.tasks.python.vision.HandLandmarker` API instead, which is why a
`.task` model file has to be downloaded once.

## Project layout

```
main.py                 # webcam loop, wires everything together
core/
  hand_geometry.py       # landmarks -> palm center, orientation, scale, finger states
  gestures.py            # declarative gesture definitions + matching
  filters.py             # One Euro Filter + angle-safe EMA smoothing
  state_machine.py        # debounced state machine + Rasengan rotation/energy tracking
  vfx_assets.py           # VFX clip loading, burst->loop playback, alpha compositing
assets/
  chidori/{chidori_burst,chidori_loop}/*.png   <- put your PNG sequence frames here
  rasengan/{rasengan_burst,rasengan_loop}/*.png
test_core_logic.py       # synthetic-data unit tests for geometry/gestures/state machine
```

## VFX asset format — read this before exporting from After Effects

**Use RGBA PNG image sequences, not WebM/MOV with alpha.**

`cv2.VideoCapture` (OpenCV's video decoder) cannot reliably surface an
alpha channel from WebM or MOV containers — alpha support depends on the
underlying FFmpeg build, and even when the video itself has alpha,
`cap.read()` only ever returns a 3-channel BGR frame. There's no supported
way to get a 4th channel out of it.

Instead, export each clip from After Effects as a **PNG sequence with
RGB + Alpha**, and drop the numbered frames into the matching folder:

```
assets/chidori/chidori_burst/0001.png, 0002.png, ...
assets/chidori/chidori_loop/0001.png, ...
assets/rasengan/rasengan_burst/0001.png, ...
assets/rasengan/rasengan_loop/0001.png, ...
```

`cv2.imread(path, cv2.IMREAD_UNCHANGED)` reliably returns true BGRA on
every platform this way. All frames are loaded into memory once at
startup, so playback is just array indexing — no per-frame disk I/O in
the render loop.

If a folder has no PNGs yet, the app doesn't crash — it renders an obvious
magenta "MISSING" placeholder clip so you can see exactly which asset slot
still needs your export.

## Gesture customization

Gestures are declared as data in `core/gestures.py`, not hardcoded
if/else logic:

```python
CHIDORI_GESTURE = GestureDefinition(
    name="chidori",
    pattern=(None, True, False, False, False),  # (thumb, index, middle, ring, pinky)
)
```

`None` means "don't care". Change the tuple to match whatever one-hand
shape you actually want to trigger Chidori. Same idea for
`RASENGAN_HAND_SHAPE`, which both hands must match, plus a distance-ratio
check (`min_dist_ratio`/`max_dist_ratio` in `recognize_rasengan_shape`)
so the hands have to be a plausible "cupping something" distance apart,
scaled by hand size so it works at any distance from the camera.

## Design choices worth knowing about

- **Palm center** = centroid of wrist + 4 MCP joints, not a single
  landmark — much less jittery than picking one point.
- **Orientation** = angle of wrist→middle-MCP vector — this joint barely
  moves when fingers curl, so rotation stays stable regardless of hand
  shape.
- **Finger extended/folded** = combination of (a) tip farther from palm
  center than PIP joint, and (b) the MCP→PIP→DIP→TIP chain being
  reasonably straight — catches fingers folded sideways that a
  distance-only check would misclassify.
- **Smoothing** = One Euro Filter for position (adapts smoothing strength
  to how fast the hand is moving — see docstring in `filters.py` for the
  full reasoning on why this was chosen over plain EMA or a Kalman
  filter), angle-wraparound-safe EMA for orientation.
- **State machine** = every gesture requires N consecutive detected
  frames to activate and M consecutive missed frames to deactivate, so a
  single dropped MediaPipe frame never causes visible flicker.
- **Rasengan rotation** = tracks the angle between the two palm centers
  over time, converts angular velocity into a decaying "energy" value
  that speeds up or slows down the loop's rotation, and gradually decays
  toward rest when the hands stop moving rather than cutting instantly.

## Tests

```bash
python test_core_logic.py
```

Runs synthetic-landmark tests for palm/orientation geometry, finger-state
classification, gesture matching (positive + negative cases), state
machine debounce/flicker resistance, Rasengan energy rise/decay, and both
smoothing filters. No webcam or model file required.
