#!/usr/bin/env python3
"""Cong cu quay du lieu EarthCam cho project dem nguoi.

Vi du:
  python3 src/capture.py list
  python3 src/capture.py snapshot --cam songridan
  python3 src/capture.py screen --cam songridan --minutes 15            # quay toan man hinh
  python3 src/capture.py screen --cam songridan --minutes 15 --pick     # click chon cua so
  python3 src/capture.py screen --cam songridan --region 0,120,1920,960 # x,y,w,h
  python3 src/capture.py stream --cam songridan --minutes 15            # doc thang luong HLS

Chay duoc tren Linux (X11) va Windows. Tren Windows dung "python" thay "python3";
voi --pick, cua so dang o tren cung khi het dem nguoc se duoc quay.

Moi phien tao 1 file .mkv va 1 file .json (thoi gian, gio dia phuong cua camera,
vung quay, fps) trong data/raw/<cam>/. Nhan Ctrl+C de dung som, file van dung duoc.
"""
import argparse
import datetime as dt
import json
import os
import platform
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

WINDOWS = platform.system() == "Windows"
if WINDOWS:
    import ctypes
    try:                                   # toa do pixel that, khong bi Windows phong to
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()

ROOT = Path(__file__).resolve().parent.parent
CAMS = json.loads((ROOT / "configs" / "cameras.json").read_text(encoding="utf-8"))
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126 Safari/537.36")
HDRS = "Referer: https://www.earthcam.com/\r\nOrigin: https://www.earthcam.com\r\n"


def stream_url(cam):
    """Lay dia chi HLS hien tai tu trang camera (token het han nen phai lay moi moi lan)."""
    c = CAMS[cam]
    req = urllib.request.Request(c["page"], headers={"User-Agent": UA})
    html = urllib.request.urlopen(req, timeout=30).read().decode("utf8", "ignore")
    m = re.search(r"var json_base\s*=\s*(\{.*?\});", html, re.S)
    if not m:
        sys.exit("Khong tim thay thong tin luong tren trang camera.")
    info = json.loads(m.group(1))["cam"][c["cam_id"]]
    if str(info.get("liveon")).lower() != "true":
        sys.exit("Camera dang offline.")
    return info["html5_streamingdomain"] + info["html5_streampath"]


def day_part(hour):
    if 6 <= hour < 11:
        return "sang"
    if 11 <= hour < 16:
        return "trua"
    if 16 <= hour < 20:
        return "chieu"
    return "dem"


def session_paths(cam, mode, tag):
    now = dt.datetime.now().astimezone()
    cam_now = now.astimezone(ZoneInfo(CAMS[cam]["timezone"]))
    name = f"{cam}_{now:%Y%m%d_%H%M%S}_{mode}_{day_part(cam_now.hour)}"
    if tag:
        name += f"_{tag}"
    d = ROOT / "data" / "raw" / cam
    d.mkdir(parents=True, exist_ok=True)
    meta = {"cam": cam, "mode": mode, "tag": tag,
            "start_local": now.isoformat(timespec="seconds"),
            "start_camera_time": cam_now.isoformat(timespec="seconds"),
            "day_part_at_camera": day_part(cam_now.hour)}
    return d / name, meta


def run_ffmpeg(cmd):
    """Chay ffmpeg; Ctrl+C gui 'q' de ffmpeg dong file dung cach."""
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        return p.wait(), False
    except KeyboardInterrupt:
        try:
            p.communicate(b"q", timeout=10)
        except Exception:
            p.terminate()
        return 0, True


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,avg_frame_rate:format=duration",
                        "-of", "json", str(path)], capture_output=True, text=True)
    try:
        j = json.loads(r.stdout)
        s = j["streams"][0]
        return {"width": s["width"], "height": s["height"], "fps": s["avg_frame_rate"],
                "duration_s": round(float(j["format"]["duration"]), 1),
                "size_mb": round(path.stat().st_size / 1e6, 1)}
    except Exception:
        return {}


def pick_region():
    """Linux: click vao cua so. Windows: lay cua so dang o tren cung (goi sau khi dem nguoc)."""
    if WINDOWS:
        from ctypes import wintypes
        u = ctypes.windll.user32
        r = wintypes.RECT()
        u.GetWindowRect(u.GetForegroundWindow(), ctypes.byref(r))
        return [r.left, r.top, r.right - r.left, r.bottom - r.top]
    print("Click vao cua so trinh duyet dang phat camera...")
    out = subprocess.run(["xwininfo"], capture_output=True, text=True).stdout
    g = lambda k: int(re.search(k + r":\s+(-?\d+)", out).group(1))
    return [g("Absolute upper-left X"), g("Absolute upper-left Y"), g("Width"), g("Height")]


def screen_size():
    if WINDOWS:
        u = ctypes.windll.user32
        return u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    out = subprocess.run(["xdpyinfo"], capture_output=True, text=True).stdout
    m = re.search(r"dimensions:\s+(\d+)x(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else (1920, 1080)


def grab_input(x, y, w, h, fps):
    """Tham so ffmpeg de quay mot vung man hinh, theo he dieu hanh."""
    if WINDOWS:
        return ["-f", "gdigrab", "-draw_mouse", "0", "-framerate", str(fps),
                "-offset_x", str(x), "-offset_y", str(y),
                "-video_size", f"{w}x{h}", "-i", "desktop"]
    return ["-f", "x11grab", "-draw_mouse", "0", "-framerate", str(fps),
            "-video_size", f"{w}x{h}", "-i", f"{os.environ.get('DISPLAY', ':0')}+{x},{y}"]


def cmd_screen(a):
    sw, sh = screen_size()
    region = None
    if a.pick and not WINDOWS:
        region = pick_region()
    elif a.region:
        region = [int(v) for v in a.region.split(",")]
    base, meta = session_paths(a.cam, "screen", a.tag)
    out = base.with_suffix(".mkv")
    hint = "dua cua so trinh duyet len tren cung" if a.pick else "dua trinh phat ve toan man hinh"
    for i in range(a.delay, 0, -1):
        print(f"\rBat dau quay sau {i}s ({hint})... ", end="", flush=True)
        time.sleep(1)
    if a.pick and WINDOWS:
        region = pick_region()
    x, y, w, h = region or [0, 0, sw, sh]
    x, y = max(0, x), max(0, y)
    w, h = min(w, sw - x) // 2 * 2, min(h, sh - y) // 2 * 2   # trong man hinh, kich thuoc chan
    print(f"\nDang quay vung {w}x{h}+{x},{y} @ {a.fps}fps trong {a.minutes} phut -> {out.name}")
    cmd = (["ffmpeg", "-hide_banner", "-loglevel", "warning", "-stats", "-y"]
           + grab_input(x, y, w, h, a.fps)
           + ["-t", str(int(a.minutes * 60)),
              "-c:v", "libx264", "-preset", "veryfast", "-crf", str(a.crf),
              "-pix_fmt", "yuv420p", str(out)])
    rc, stopped = run_ffmpeg(cmd)
    meta.update({"os": platform.system(), "region_xywh": [x, y, w, h], "fps_requested": a.fps,
                 "crf": a.crf, "screen": [sw, sh], "stopped_early": stopped,
                 "files": {out.name: probe(out)}})
    base.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Xong:", json.dumps(meta["files"], ensure_ascii=False))
    return rc


def cmd_stream(a):
    base, meta = session_paths(a.cam, "stream", a.tag)
    total = a.minutes * 60
    t0, part, files, stopped = time.time(), 0, {}, False
    while not stopped and total - (time.time() - t0) > 3 and part < 20:
        part += 1
        out = Path(f"{base}_p{part:02d}.mkv")
        remain = int(total - (time.time() - t0))
        print(f"Phan {part}: ghi {remain}s -> {out.name}")
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-stats", "-y",
               "-user_agent", UA, "-headers", HDRS, "-i", stream_url(a.cam),
               "-t", str(remain), "-c", "copy", str(out)]
        rc, stopped = run_ffmpeg(cmd)
        if out.exists() and out.stat().st_size > 0:
            files[out.name] = probe(out)
        elif not stopped:
            time.sleep(5)          # luong loi, cho roi lay token moi
    meta.update({"stopped_early": stopped, "files": files})
    base.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Xong:", json.dumps(files, ensure_ascii=False))


def cmd_snapshot(a):
    d = ROOT / "data" / "snapshots"
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{a.cam}_{dt.datetime.now():%Y%m%d_%H%M%S}.jpg"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-user_agent", UA,
                    "-headers", HDRS, "-i", stream_url(a.cam), "-frames:v", "1", "-q:v", "2",
                    str(out)], check=True)
    print(out)


def cmd_list(a):
    now = dt.datetime.now().astimezone()
    for k, c in CAMS.items():
        t = now.astimezone(ZoneInfo(c["timezone"]))
        print(f"{k:10s} [{c['role']}] {c['title']} | gio tai camera: {t:%H:%M} ({day_part(t.hour)})")
        print(f"{'':10s} {c['page']}")
        d = ROOT / "data" / "raw" / k
        for j in sorted(d.glob("*.json")) if d.exists() else []:
            m = json.loads(j.read_text(encoding="utf-8"))
            dur = sum(f.get("duration_s", 0) for f in m["files"].values())
            print(f"{'':12s}- {j.stem}: {dur/60:.1f} phut, {m['day_part_at_camera']}")


def main():
    import shutil
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        sys.exit("Chua co ffmpeg/ffprobe trong PATH. Windows: winget install Gyan.FFmpeg ; Ubuntu: sudo apt install ffmpeg")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for name, fn in (("screen", cmd_screen), ("stream", cmd_stream), ("snapshot", cmd_snapshot)):
        s = sub.add_parser(name)
        s.add_argument("--cam", required=True, choices=list(CAMS))
        s.set_defaults(fn=fn)
        if name != "snapshot":
            s.add_argument("--minutes", type=float, default=15)
            s.add_argument("--tag", default="", help="ghi chu them, vi du: mua, cuoituan")
        if name == "screen":
            s.add_argument("--fps", type=int, default=25)
            s.add_argument("--crf", type=int, default=20)
            s.add_argument("--region", help="x,y,w,h")
            s.add_argument("--pick", action="store_true", help="click chon cua so")
            s.add_argument("--delay", type=int, default=5, help="so giay dem nguoc truoc khi quay")
    sub.add_parser("list").set_defaults(fn=cmd_list)
    a = ap.parse_args()
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()
