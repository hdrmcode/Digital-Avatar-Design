"""
Digital Avatar Design - real-time flat-vector avatar (v8)
Run:  .\venv\Scripts\python.exe avatar.py
Keys: a auto colours | s skin | k hair colour | h hairstyle | f facial hair | e eyes
      t shirt | b background | m mesh | r recalibrate | c capture | q quit
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GLOG_minloglevel"] = "2"
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import math
import time

import cv2
import numpy as np
import mediapipe as mp

CAMERA_INDEX = 0      # 0 = w200
AV = 480              # avatar size in pixels
SS = 2                # supersampling factor for smooth edges
PANEL_W = 270
BLINK_EAR = 0.18
CALIB_FRAMES = 30
HEAD_SCALE = 1.06     # overall head size
AA = cv2.LINE_AA

# Curated colour palettes (BGR)
SKIN = [(205, 222, 250), (180, 205, 245), (150, 185, 232), (120, 160, 215),
        (95, 135, 190), (75, 110, 165), (60, 88, 135), (48, 68, 105)]          # light -> deep
HAIR_COLS = [(35, 30, 32), (40, 55, 85), (45, 80, 130), (40, 70, 150),
             (110, 170, 215), (160, 160, 165)]                                # black, dark brown, brown, auburn, blonde, grey
EYES = [(40, 75, 120), (215, 150, 50), (100, 150, 80), (30, 45, 70)]          # brown, blue, green, dark
SHIRTS = [(110, 60, 35), (75, 62, 52), (160, 120, 80), (60, 70, 160), (110, 140, 120)]
BGS = [(243, 243, 243), (245, 235, 225), (225, 235, 245), (70, 60, 55)]
HAIRSTYLES = ["curly", "short", "side part", "long", "buzz"]
FACIAL = ["none", "stubble", "beard"]

# One Euro filter settings per feature: (min_cutoff, beta)
FILTER_PARAMS = {"ear_l": (4.0, 0.5), "ear_r": (4.0, 0.5), "mar": (4.0, 0.5),
                 "smile_w": (1.5, 0.5), "brow": (1.5, 0.5), "roll": (1.0, 0.05),
                 "yaw": (1.0, 1.0), "pitch": (1.0, 1.0), "aspect": (0.5, 0.0)}

FM = mp.solutions.face_mesh
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
GAMMA_LUT = np.array([((i / 255.0) ** 0.65) * 255 for i in range(256)], np.uint8)


# ---------------------------------------------------------------- helpers
class OneEuro:
    """One Euro filter: smooth when still, responsive when moving."""
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.mc, self.beta, self.dc = min_cutoff, beta, d_cutoff
        self.x, self.dx, self.t = None, 0.0, None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.t = x, t
            return x
        dt = max(t - self.t, 1e-3)
        self.t = t
        a_d = self._alpha(self.dc, dt)
        self.dx = a_d * ((x - self.x) / dt) + (1 - a_d) * self.dx
        a = self._alpha(self.mc + self.beta * abs(self.dx), dt)
        self.x = a * x + (1 - a) * self.x
        return self.x


def enhance(frame):
    """Brighten dark frames and boost local contrast to help face tracking."""
    if frame.mean() < 100:
        frame = cv2.LUT(frame, GAMMA_LUT)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = CLAHE.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def shade(c, f):
    return tuple(int(np.clip(v * f, 0, 255)) for v in c)


def tint(c, d):
    return tuple(int(np.clip(v + d, 0, 255)) for v in c)


def mix(a, b, t):
    return tuple(int(x * (1 - t) + y * t) for x, y in zip(a, b))


def P(x, y):
    return (int(round(x * SS)), int(round(y * SS)))


def A(a, b):
    return (max(1, int(round(a * SS))), max(1, int(round(b * SS))))


def L(x, y, off):
    """Point in a local sub-image whose top-left corner is `off`."""
    return (int(round(x * SS)) - int(off[0]), int(round(y * SS)) - int(off[1]))


def pts2(a):
    return np.round(np.asarray(a, np.float64) * SS).astype(np.int32)


def V(*parts):
    return np.vstack([np.asarray(p, np.float64).reshape(-1, 2) for p in parts])


def arc(cx, cy, ax, ay, a0, a1, n=24):
    t = np.radians(np.linspace(a0, a1, n))
    return np.stack([cx + ax * np.cos(t), cy + ay * np.sin(t)], 1)


def bezier(p0, p1, p2, p3, n=16):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = [np.array(p, np.float64) for p in (p0, p1, p2, p3)]
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t * t * p2 + (t ** 3) * p3


def almond(ex, ey, hw, eh, n=14):
    u = np.linspace(0, 1, n)
    xs = ex - hw + 2 * hw * u
    upper = np.stack([xs, ey - eh * np.sin(np.pi * u)], 1)
    lower = np.stack([xs[::-1], ey + 0.6 * eh * np.sin(np.pi * u[::-1])], 1)
    return upper, np.vstack([upper, lower])


def mouth_curve(cx, my, mw, lift, sag, n=18):
    u = np.linspace(0, 1, n)
    s = np.sin(np.pi * u)
    return np.stack([cx - mw + 2 * mw * u, my - lift * (1 - s) + sag * s], 1)


def brow_shape(ex, by, side):
    inner, outer = ex - side * 20, ex + side * 25
    top = bezier((inner, by), (ex - side * 8, by - 8), (ex + side * 12, by - 9), (outer, by + 1), 12)
    bot = bezier((outer, by + 3), (ex + side * 12, by - 3), (ex - side * 8, by - 1), (inner, by + 8), 12)
    return np.vstack([top, bot])


def draw_in_region(canvas, region, draw_fn):
    """Run draw_fn on a copy of the region's bounding box, keep pixels inside the region only."""
    pp = pts2(region)
    size = canvas.shape[0]
    x0, y0 = [max(int(v) - 4, 0) for v in pp.min(axis=0)]
    x1, y1 = [min(int(v) + 4, size) for v in pp.max(axis=0)]
    if x1 <= x0 or y1 <= y0:
        return
    sub = canvas[y0:y1, x0:x1]
    lay = sub.copy()
    draw_fn(lay, (x0, y0))
    m = np.zeros(sub.shape[:2], np.uint8)
    cv2.fillPoly(m, [pp - np.array([x0, y0], np.int32)], 255)
    sub[m > 0] = lay[m > 0]


def tint_region(canvas, region, color, alpha):
    """Blend a colour over a polygon region (used for stubble)."""
    def blend(lay, off):
        over = np.empty_like(lay)
        over[:] = color
        lay[:] = cv2.addWeighted(over, alpha, lay, 1 - alpha, 0)
    draw_in_region(canvas, region, blend)


def pill(img, x1, y1, x2, y2, color):
    r = (y2 - y1) // 2
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, AA)
    cv2.circle(img, (x1 + r, y1 + r), r, color, -1, AA)
    cv2.circle(img, (x2 - r, y1 + r), r, color, -1, AA)


def label(img, text):
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, _), _ = cv2.getTextSize(text, font, 0.45, 1)
    pill(img, 12, 12, 12 + tw + 28, 40, (35, 32, 30))
    cv2.putText(img, text, (26, 31), font, 0.45, (240, 240, 240), 1, AA)


def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


# ------------------------------------------------- personalised colours
def sample_patch(frame, x, y, r=6):
    h, w = frame.shape[:2]
    x, y = int(x), int(y)
    patch = frame[max(y - r, 0):min(y + r, h), max(x - r, 0):min(x + r, w)]
    if patch.size == 0:
        return None
    return np.median(patch.reshape(-1, 3), axis=0)


def auto_colors(frame, lm):
    """Match the person's skin and hair to the nearest natural palette colour."""
    h, w = frame.shape[:2]
    p = lambda i: np.array([lm[i].x * w, lm[i].y * h])
    skin_i = hair_i = None

    samples = [sample_patch(frame, *p(i)) for i in (50, 280, 151)]
    samples = [s for s in samples if s is not None]
    if samples:
        px = np.uint8([[np.median(samples, axis=0)]])
        lightness = float(cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0, 0])
        skin_i = int(np.clip(round((205 - lightness) / 17), 0, len(SKIN) - 1))

    hair_raw = sample_patch(frame, *(p(10) + (p(10) - p(152)) * 0.15), r=8)
    if hair_raw is not None:
        px = np.uint8([[hair_raw]])
        lh = float(cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0, 0])
        hue, sat, _ = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0]
        if lh < 75:
            hair_i = 0          # black
        elif sat < 40 and lh > 140:
            hair_i = 5          # grey
        elif lh > 150 and 10 <= hue <= 30:
            hair_i = 4          # blonde
        elif lh < 105:
            hair_i = 1          # dark brown
        elif hue < 10:
            hair_i = 3          # auburn
        else:
            hair_i = 2          # brown
    return skin_i, hair_i


# ------------------------------------------------------- feature extraction
def extract_features(lm, w, h):
    """Expression and pose values. Left/right are decided by screen position."""
    p = lambda i: (lm[i].x * w, lm[i].y * h)
    eps = 1e-6
    ear = lambda up, lo, c1, c2: dist(p(up), p(lo)) / (dist(p(c1), p(c2)) + eps)
    ear_a = ear(159, 145, 33, 133)
    ear_b = ear(386, 374, 362, 263)
    if p(33)[0] < p(263)[0]:
        ear_l, ear_r = ear_a, ear_b
    else:
        ear_l, ear_r = ear_b, ear_a
    mar = dist(p(13), p(14)) / (dist(p(78), p(308)) + eps)
    face_h = dist(p(10), p(152)) + eps
    face_w = dist(p(234), p(454)) + eps
    brow = (dist(p(334), p(386)) + dist(p(105), p(159))) / 2 / face_h
    smile_w = dist(p(61), p(291)) / face_w
    (lx, ly), (rx, ry) = sorted([p(33), p(263)])
    roll = math.degrees(math.atan2(ry - ly, rx - lx))
    cl, cr = sorted([p(234)[0], p(454)[0]])
    yaw = ((p(1)[0] - cl) / (cr - cl + eps) - 0.5) * 2
    pitch = ((p(1)[1] - p(10)[1]) / face_h - 0.5) * 2
    return dict(ear_l=ear_l, ear_r=ear_r, mar=mar, brow=brow, smile_w=smile_w,
                aspect=face_w / face_h, roll=roll, yaw=yaw, pitch=pitch)


def levels(f, base):
    eye = lambda e: float(np.clip((e - 0.13) / (0.28 - 0.13), 0, 1))
    roll = f["roll"]
    return dict(
        eye_l=eye(f["ear_l"]),
        eye_r=eye(f["ear_r"]),
        mouth=float(np.clip((f["mar"] - 0.05) / 0.5, 0, 1)),
        smile=float(np.clip((f["smile_w"] - base["smile_w"]) / 0.04, 0, 1)),
        brow=float(np.clip((f["brow"] - base["brow"]) * 900, -6, 22)),
        roll=0.0 if abs(roll) < 3 else float(np.clip(roll, -25, 25)),
        yaw=float(np.clip(f["yaw"], -0.7, 0.7)),
        pitch=float(np.clip(f["pitch"] - base["pitch"], -0.5, 0.5)),
    )


# ------------------------------------------------------------ the avatar
def draw_avatar(lv, look, hairstyle, facial, face_w):
    S = AV * SS
    skin, hair, eye_c, shirt = look["skin"], look["hair"], look["eye"], look["shirt"]
    img = np.full((S, S, 3), look["bg"], np.uint8)

    # ---- Body: short wide neck, subtle chin shadow, collared shirt (does not rotate)
    neck_c = shade(skin, 0.9)
    cv2.rectangle(img, P(189, 300), P(291, 430), neck_c, -1)
    cv2.ellipse(img, P(240, 352), A(46, 11), 0, 0, 360, shade(skin, 0.8), -1, AA)
    cv2.ellipse(img, P(240, 535), A(238, 165), 0, 180, 360, shirt, -1, AA)
    cv2.fillPoly(img, [pts2([(203, 369), (240, 423), (277, 369)])], neck_c, AA)
    for side in (-1, 1):
        flap = [(240 - side * 38, 363), (240 - side * 68, 377), (240 - side * 46, 427), (240 - side * 2, 419)]
        cv2.fillPoly(img, [pts2([(x, y + 5) for x, y in flap])], shade(shirt, 0.78), AA)
        cv2.fillPoly(img, [pts2(flap)], tint(shirt, 35), AA)
    cv2.line(img, P(240, 423), P(240, 480), shade(shirt, 0.8), SS, AA)
    for y in (441, 461):
        cv2.circle(img, P(240, y), int(2.6 * SS), (225, 225, 225), -1, AA)

    # ---- Head layer (drawn separately so it can tilt and scale)
    head = np.zeros_like(img)
    mask = np.zeros((S, S), np.uint8)
    W = face_w
    hx = 240 + lv["yaw"] * 8
    cx, cy = hx + lv["yaw"] * 32, 205 + lv["pitch"] * 22

    def fill(points, color, to_mask=True):
        pp = pts2(points)
        cv2.fillPoly(head, [pp], color, AA)
        if to_mask:
            cv2.fillPoly(mask, [pp], 255, AA)

    def circ(c, r, color, to_mask=True):
        cv2.circle(head, P(*c), int(r * SS), color, -1, AA)
        if to_mask:
            cv2.circle(mask, P(*c), int(r * SS), 255, -1, AA)

    def ell(c, ax, color, to_mask=True):
        cv2.ellipse(head, P(*c), A(*ax), 0, 0, 360, color, -1, AA)
        if to_mask:
            cv2.ellipse(mask, P(*c), A(*ax), 0, 0, 360, 255, -1, AA)

    hair_c = mix(hair, skin, 0.45) if hairstyle == "buzz" else hair

    # ---- Hair (back)
    if hairstyle == "curly":
        fill(V(arc(hx, 198, W + 18, 148, 180, 360), [(hx + W + 18, 205), (hx - W - 18, 205)]), hair_c)
        for a in range(188, 356, 15):
            t = math.radians(a)
            circ((hx + (W + 16) * math.cos(t), 198 + 144 * math.sin(t)), 23, hair_c)
    elif hairstyle == "long":
        fill(V(arc(hx, 198, W + 18, 146, 180, 360), [(hx + W + 18, 205), (hx - W - 18, 205)]), hair_c)
        for side in (-1, 1):
            curtain = [(hx + side * (W + 18), 190), (hx + side * (W + 30), 300), (hx + side * (W + 26), 410),
                       (hx + side * (W - 28), 412), (hx + side * (W - 30), 330), (hx + side * (W - 6), 260)]
            fill(curtain, hair_c)
    elif hairstyle == "buzz":
        fill(V(arc(hx, 200, W + 4, 132, 180, 360), [(hx + W + 4, 205), (hx - W - 4, 205)]), hair_c)
    else:   # short, side part
        fill(V(arc(hx, 200, W + 11, 141, 180, 360), [(hx + W + 11, 215), (hx - W - 11, 215)]), hair_c)

    # ---- Ears
    for side in (-1, 1):
        ell((hx + side * (W + 3), 218), (16, 29), skin)
        cv2.ellipse(head, P(hx + side * (W + 5), 218), A(8, 17), 0, 0, 360, shade(skin, 0.88), -1, AA)

    # ---- Face shape (rounded forehead, tapered jaw)
    right_jaw = bezier((hx + W, 200), (hx + W, 292), (hx + 0.55 * W, 347), (hx, 349), 20)
    left_jaw = bezier((hx, 349), (hx - 0.55 * W, 347), (hx - W, 292), (hx - W, 200), 20)
    fill(V(arc(hx, 200, W, 125, 180, 360), right_jaw, left_jaw), skin)

    # Soft shading along the right jaw (flat 2-tone style)
    t = np.linspace(0, 1, len(right_jaw))[:, None]
    dirv = np.array([hx, 260.0]) - right_jaw
    dirv /= np.linalg.norm(dirv, axis=1, keepdims=True) + 1e-6
    inner = right_jaw + dirv * (14 * np.sin(np.pi * t))
    fill(V(right_jaw, inner[::-1]), shade(skin, 0.93), to_mask=False)

    # ---- Facial hair
    if facial != "none":
        jaw = V(bezier((hx - W, 230), (hx - W, 292), (hx - 0.55 * W, 347), (hx, 349)),
                bezier((hx, 349), (hx + 0.55 * W, 347), (hx + W, 292), (hx + W, 230)))
        top_edge = V(bezier((hx + W, 230), (hx + 0.8 * W, cy + 72), (hx + 0.35 * W, cy + 60), (hx, cy + 63)),
                     bezier((hx, cy + 63), (hx - 0.35 * W, cy + 60), (hx - 0.8 * W, cy + 72), (hx - W, 230)))
        region = V(jaw, top_edge)
        if facial == "stubble":
            tint_region(head, region, hair, 0.3)
        else:
            fill(region, shade(hair, 0.95), to_mask=False)

    # ---- Eyes and eyebrows
    dark = (55, 45, 45)
    for side, openness in ((-1, lv["eye_l"]), (1, lv["eye_r"])):
        ex, ey = cx + side * 48, cy - 5
        eh = 13 * openness
        if eh < 3:
            cv2.ellipse(head, P(ex, ey), A(21, 6), 0, 15, 165, dark, 2 * SS, AA)
        else:
            upper, eye_poly = almond(ex, ey, 23, eh)
            cv2.fillPoly(head, [pts2(eye_poly)], (250, 250, 250), AA)
            ix, iy = ex + lv["yaw"] * 8, ey + 1 + lv["pitch"] * 3

            def iris(im, off, ix=ix, iy=iy):
                cv2.circle(im, L(ix, iy, off), int(11 * SS), eye_c, -1, AA)
                cv2.circle(im, L(ix, iy, off), int(11 * SS), shade(eye_c, 0.7), int(1.5 * SS), AA)
                cv2.circle(im, L(ix, iy, off), int(5 * SS), (25, 25, 25), -1, AA)
                cv2.circle(im, L(ix + 3.5, iy - 3.5, off), int(2.5 * SS), (255, 255, 255), -1, AA)
            draw_in_region(head, eye_poly, iris)
            cv2.polylines(head, [pts2(upper)], False, dark, 2 * SS, AA)
        by = ey - 27 - lv["brow"] * 0.8
        cv2.fillPoly(head, [pts2(brow_shape(ex, by, side))], shade(hair, 0.85), AA)

    # ---- Nose
    nc = shade(skin, 0.82)
    cv2.polylines(head, [pts2([(cx + 4, cy + 5), (cx + 10, cy + 36), (cx + 9, cy + 44), (cx + 1, cy + 47)])],
                  False, nc, int(2.5 * SS), AA)
    cv2.ellipse(head, P(cx - 7, cy + 44), A(4, 2.5), 0, 0, 360, nc, -1, AA)

    # ---- Mouth: open / smiling with teeth / neutral
    my = cy + 80
    smile, opened = lv["smile"], lv["mouth"]
    mw = 24 + 7 * smile
    lift = 2 + 5 * smile
    line_c = shade(skin, 0.62)
    if opened > 0.12:
        region = V(mouth_curve(cx, my, mw, lift, -1), mouth_curve(cx, my, mw, lift, 6 + 26 * opened)[::-1])
        cv2.fillPoly(head, [pts2(region)], (60, 40, 95), AA)

        def inside(im, off):
            cv2.rectangle(im, L(cx - mw, my - 10, off), L(cx + mw, my + 4, off), (248, 248, 248), -1)
            cv2.ellipse(im, L(cx, my + 6 + 26 * opened, off), A(mw * 0.55, 5 + 8 * opened),
                        0, 0, 360, (140, 130, 230), -1, AA)
        draw_in_region(head, region, inside)
        cv2.polylines(head, [pts2(region)], True, line_c, int(1.5 * SS), AA)
    elif smile > 0.35:
        region = V(mouth_curve(cx, my, mw, lift, 0), mouth_curve(cx, my, mw, lift, 5 + 8 * smile)[::-1])
        cv2.fillPoly(head, [pts2(region)], (250, 250, 250), AA)
        cv2.polylines(head, [pts2(region)], True, line_c, int(1.5 * SS), AA)
        cv2.ellipse(head, P(cx, my + 12 + 8 * smile), A(11, 3), 0, 20, 160, shade(skin, 0.86), 2 * SS, AA)
    else:
        cv2.polylines(head, [pts2(mouth_curve(cx, my, mw * 0.9, lift, 2))], False, line_c, int(2.5 * SS), AA)
        cv2.ellipse(head, P(cx, my + 10), A(11, 3), 0, 20, 160, shade(skin, 0.86), 2 * SS, AA)

    # ---- Hair (front)
    if hairstyle == "curly":
        hairline = V([(hx + W + 12, 205), (hx + W, 205), (hx + W - 4, 150)],
                     bezier((hx + W - 6, 140), (hx + 0.4 * W, 118), (hx - 0.4 * W, 118), (hx - W + 6, 140)),
                     [(hx - W + 4, 150), (hx - W, 205), (hx - W - 12, 205)])
        fill(V(arc(hx, 200, W + 12, 142, 180, 360), hairline), hair_c)
        for i, x in enumerate(np.linspace(hx - W + 12, hx + W - 12, 9)):
            circ((x, 124 + 3 * (i % 2)), 13, hair_c)
        for side in (-1, 1):
            circ((hx + side * (W - 2), 150), 13, hair_c)
    elif hairstyle == "short":
        hl = [(hx + W + 10, 200), (hx + W + 2, 200), (hx + W - 2, 160), (hx + W - 14, 128)]
        for i, x in enumerate(np.linspace(hx + W - 26, hx - W + 26, 9)):
            hl.append((x, 116 + (12 if i % 2 == 0 else 0) - (x - hx) * 0.04))
        hl += [(hx - W + 14, 128), (hx - W + 2, 160), (hx - W - 2, 200), (hx - W - 10, 200)]
        fill(V(arc(hx, 200, W + 10, 140, 180, 360), hl), hair_c)
    elif hairstyle == "side part":
        hairline = V([(hx + W + 10, 200), (hx + W + 2, 200), (hx + W - 2, 155)],
                     bezier((hx + W - 4, 145), (hx + 0.5 * W, 105), (hx - 0.2 * W, 120), (hx - W + 2, 158)),
                     [(hx - W - 2, 200), (hx - W - 10, 200)])
        fill(V(arc(hx, 200, W + 10, 140, 180, 360), hairline), hair_c)
        for k, dv in ((0, 30), (8, 20)):
            strand = bezier((hx + 0.35 * W, 72 + k), (hx + 0.3 * W, 90 + k),
                            (hx + 0.1 * W, 100 + k), (hx - 0.25 * W, 118 + k))
            cv2.polylines(head, [pts2(strand)], False, tint(hair_c, dv), int(2.5 * SS), AA)
    elif hairstyle == "long":
        for side in (-1, 1):
            curtain = V(bezier((hx - side * 2, 68), (hx + side * 0.5 * W, 70),
                               (hx + side * (W + 6), 110), (hx + side * (W + 6), 215)),
                        bezier((hx + side * (W - 6), 215), (hx + side * (W - 4), 150),
                               (hx + side * 0.5 * W, 108), (hx + side * 2, 96)))
            fill(curtain, hair_c)
    else:   # buzz
        hairline = V([(hx + W + 3, 200), (hx + W - 1, 200), (hx + W - 4, 150)],
                     bezier((hx + W - 8, 135), (hx + 0.4 * W, 104), (hx - 0.4 * W, 104), (hx - W + 8, 135)),
                     [(hx - W + 4, 150), (hx - W + 1, 200), (hx - W - 3, 200)])
        fill(V(arc(hx, 200, W + 3, 131, 180, 360), hairline), hair_c)

    # ---- Tilt and scale the head around the neck, then place it on the body
    M = cv2.getRotationMatrix2D((hx * SS, 340 * SS), -lv["roll"], HEAD_SCALE)
    head = cv2.warpAffine(head, M, (S, S))
    mask = cv2.warpAffine(mask, M, (S, S))
    m = mask > 127
    img[m] = head[m]
    return cv2.resize(img, (AV, AV), interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------- info panel
def draw_panel(status, calib_n, fps, blinks, lv, hairstyle, facial):
    p = np.full((AV, PANEL_W, 3), (32, 30, 28), np.uint8)
    duplex, simplex = cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_SIMPLEX

    def text(s, x, y, scale=0.5, col=(230, 230, 230), font=duplex):
        cv2.putText(p, s, (x, y), font, scale, col, 1, AA)

    def bar(name, y, value, color):
        text(name, 18, y, 0.42, (175, 175, 175))
        cv2.rectangle(p, (18, y + 7), (PANEL_W - 18, y + 14), (62, 58, 55), -1)
        x2 = 18 + int((PANEL_W - 36) * float(np.clip(value, 0, 1)))
        if x2 > 18:
            cv2.rectangle(p, (18, y + 7), (x2, y + 14), color, -1)

    text("DIGITAL AVATAR", 18, 38, 0.72, (255, 255, 255))
    text("Metaverse Technologies", 18, 60, 0.42, (150, 150, 150))

    colors = {"Tracking": (95, 185, 95), "Calibrating": (60, 165, 235), "No face": (85, 85, 225)}
    label_txt = {"Tracking": "TRACKING", "No face": "NO FACE DETECTED",
                 "Calibrating": f"CALIBRATING {calib_n}/{CALIB_FRAMES}"}[status]
    pill(p, 18, 78, PANEL_W - 18, 106, colors[status])
    (tw, _), _ = cv2.getTextSize(label_txt, duplex, 0.45, 1)
    text(label_txt, (PANEL_W - tw) // 2, 97, 0.45, (25, 25, 25))

    text("FPS", 18, 138, 0.42, (150, 150, 150))
    text(f"{fps:.0f}", 18, 170, 0.95, (255, 255, 255))
    text("BLINKS", 145, 138, 0.42, (150, 150, 150))
    text(str(blinks), 145, 170, 0.95, (255, 255, 255))

    bar("Left eye", 198, lv["eye_l"], (235, 175, 90))
    bar("Right eye", 230, lv["eye_r"], (235, 175, 90))
    bar("Mouth", 262, lv["mouth"], (130, 120, 240))
    bar("Smile", 294, lv["smile"], (120, 200, 120))
    bar("Eyebrows", 326, (lv["brow"] + 6) / 28, (90, 200, 220))

    text(f"Head tilt  {lv['roll']:+.0f} deg   turn {lv['yaw']:+.2f}", 18, 365, 0.42)
    text(f"Hair: {hairstyle}   Facial: {facial}", 18, 388, 0.42)

    help_col = (140, 140, 140)
    text("A auto colours  S skin  K hair colour", 18, 420, 0.38, help_col, simplex)
    text("H hairstyle  F facial hair  E eyes", 18, 437, 0.38, help_col, simplex)
    text("T shirt  B background  M mesh", 18, 454, 0.38, help_col, simplex)
    text("R recalibrate  C capture  Q quit", 18, 471, 0.38, help_col, simplex)
    return p


# -------------------------------------------------------------------- main
def main():
    cap = open_camera()
    if not cap.isOpened():
        raise SystemExit("Could not open webcam - check camera privacy settings")

    mesh = FM.FaceMesh(max_num_faces=1, refine_landmarks=True,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    filters = {k: OneEuro(*v) for k, v in FILTER_PARAMS.items()}
    lists = {"skin": SKIN, "hair": HAIR_COLS, "eye": EYES, "shirt": SHIRTS, "bg": BGS}
    idx = {"skin": 3, "hair": 0, "eye": 0, "shirt": 0, "bg": 0}
    look = {k: lists[k][i] for k, i in idx.items()}
    hair_i, facial_i = 0, 0
    show_mesh = True
    base = dict(brow=0.07, pitch=0.1, smile_w=0.42, aspect=0.9)
    calib, calibrated, auto_done = [], False, False
    feats = dict(ear_l=0.28, ear_r=0.28, mar=0.0, brow=0.07, smile_w=0.42,
                 aspect=0.9, roll=0.0, yaw=0.0, pitch=0.1)
    last_clean, last_lm = None, None
    blinks, eyes_closed = 0, False
    fps, t_prev = 0.0, time.time()
    failed_reads = 0

    def apply_auto():
        skin_i, hair_c_i = auto_colors(last_clean, last_lm)
        if skin_i is not None:
            idx["skin"] = skin_i
            look["skin"] = SKIN[skin_i]
        if hair_c_i is not None:
            idx["hair"] = hair_c_i
            look["hair"] = HAIR_COLS[hair_c_i]

    try:
        import pyvirtualcam
        vcam = pyvirtualcam.Camera(width=AV, height=AV, fps=30)
        print("Virtual camera started:", vcam.device)
    except Exception:
        vcam = None

    window = "Digital Avatar"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    print("Running - press q in the window to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            failed_reads += 1
            if failed_reads > 30:
                print("Could not read frames from the webcam")
                break
            continue
        failed_reads = 0

        frame = enhance(cv2.flip(frame, 1))
        h, w = frame.shape[:2]
        result = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face_found = bool(result.multi_face_landmarks)
        now = time.time()

        if face_found:
            face = result.multi_face_landmarks[0]
            last_clean, last_lm = frame.copy(), face.landmark
            raw = extract_features(face.landmark, w, h)

            avg_ear = (raw["ear_l"] + raw["ear_r"]) / 2
            if avg_ear < BLINK_EAR and not eyes_closed:
                blinks, eyes_closed = blinks + 1, True
            elif avg_ear >= BLINK_EAR:
                eyes_closed = False

            if not calibrated:
                calib.append(raw)
                if len(calib) >= CALIB_FRAMES:
                    base = {k: float(np.mean([c[k] for c in calib])) for k in base}
                    calibrated = True

            if calibrated and not auto_done:          # personalise colours once
                apply_auto()
                auto_done = True

            feats = {k: filters[k](v, now) for k, v in raw.items()}

            if show_mesh:
                mp_draw.draw_landmarks(
                    image=frame, landmark_list=face,
                    connections=FM.FACEMESH_TESSELATION, landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style())

        lv = levels(feats, base)
        face_w = float(np.clip(118 * base["aspect"] / 0.9, 106, 130))
        avatar = draw_avatar(lv, look, HAIRSTYLES[hair_i], FACIAL[facial_i], face_w)

        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now

        if not calibrated:
            status = "Calibrating"
        else:
            status = "Tracking" if face_found else "No face"
        panel = draw_panel(status, len(calib), fps, blinks, lv, HAIRSTYLES[hair_i], FACIAL[facial_i])

        cam_view = cv2.resize(frame, (int(w * AV / h), AV))
        label(cam_view, "CAMERA")
        av_view = avatar.copy()
        label(av_view, "AVATAR")
        sep = np.full((AV, 2, 3), 20, np.uint8)
        combined = np.hstack([cam_view, sep, av_view, sep, panel])
        cv2.imshow(window, combined)

        if vcam:
            vcam.send(cv2.cvtColor(avatar, cv2.COLOR_BGR2RGB))

        key = cv2.waitKey(1) & 0xFF
        cycle = {ord("s"): "skin", ord("k"): "hair", ord("e"): "eye", ord("t"): "shirt", ord("b"): "bg"}
        if key == ord("q"):
            break
        elif key in cycle:
            name = cycle[key]
            idx[name] = (idx[name] + 1) % len(lists[name])
            look[name] = lists[name][idx[name]]
        elif key == ord("a") and last_lm is not None:
            apply_auto()
        elif key == ord("h"):
            hair_i = (hair_i + 1) % len(HAIRSTYLES)
        elif key == ord("f"):
            facial_i = (facial_i + 1) % len(FACIAL)
        elif key == ord("m"):
            show_mesh = not show_mesh
        elif key == ord("r"):
            calib, calibrated, auto_done = [], False, False
        elif key == ord("c"):
            fname = time.strftime("avatar_%Y%m%d_%H%M%S.png")
            cv2.imwrite(fname, combined)
            print("Saved", fname)

    cap.release()
    cv2.destroyAllWindows()
    if vcam:
        vcam.close()


if __name__ == "__main__":
    main()