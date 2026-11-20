#!/usr/bin/env python3
"""Blender camera move -> Wan 2.1 VACE 1.3B video, on a 6 GB GPU.

  python make_video.py --blend scene.blend --prompt "..." --seconds 5 --fps 24

Steps: Blender renders a depth pass of your scene -> uploaded to ComfyUI ->
Wan 2.1 VACE 1.3B follows that depth (context windows for clips > 5 s) ->
optional RIFE interpolation 16 -> 24 fps -> MP4 in outputs/<name>/.
"""
import argparse
import datetime as dt
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
WAN_FPS = 16
WINDOW = 81  # Wan 2.1 training length; longer clips use context windows

DEFAULT_MODELS = {
    "unet": "wan2.1_vace_1.3B_fp16.safetensors",
    "clip": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "vae": "wan_2.1_vae.safetensors",
    "rife": "rife_v4.26.safetensors",
}
DEFAULT_NEGATIVE = "blurry, low quality, distorted, overexposed, watermark, text, deformed, gray, untextured"


# ---------------------------------------------------------------- helpers

def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def wan_frame_count(seconds):
    """16 fps frame count rounded to the 4n+1 grid Wan needs."""
    return max(5, 4 * round(seconds * WAN_FPS / 4) + 1)


def load_config():
    """config.json written by setup.py (paths found on this machine)."""
    try:
        return json.loads((HERE / "config.json").read_text())
    except (OSError, ValueError):
        return {}


CONFIG = load_config()


def find_blender(explicit, required=True):
    candidates = [explicit, CONFIG.get("blender"), os.environ.get("BLENDER"), shutil.which("blender")]
    patterns = [
        r"C:\Program Files\Blender Foundation\Blender*\blender.exe",
        r"C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe",
        r"C:\Program Files\Steam\steamapps\common\Blender\blender.exe",
        "/Applications/Blender.app/Contents/MacOS/Blender",
        os.path.expanduser("~/.steam/steam/steamapps/common/Blender/blender"),
        "/usr/bin/blender", "/snap/bin/blender",
    ]
    for pat in patterns:
        candidates += sorted(glob.glob(pat), reverse=True)  # newest version first
    for c in candidates:
        if c and Path(c).is_file():
            return str(c)
    if required:
        die("Blender not found. Run setup again, pass --blender PATH or set the BLENDER environment variable.")
    return None


# ---------------------------------------------------------------- ComfyUI API

class Comfy:
    def __init__(self, server):
        self.server = server.rstrip("/")

    def request(self, path, data=None, headers=None, raw=False, timeout=60):
        req = urllib.request.Request(self.server + path, data=data, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
        except urllib.error.HTTPError as e:
            die(f"ComfyUI {path} -> HTTP {e.code}: {e.read().decode(errors='replace')[:3000]}")
        except urllib.error.URLError as e:
            die(f"cannot reach ComfyUI at {self.server} ({e.reason}). Is it running?")
        return body if raw else json.loads(body)

    def post_json(self, path, payload):
        return self.request(path, json.dumps(payload).encode(), {"Content-Type": "application/json"})

    def upload(self, file_path, name):
        boundary = uuid.uuid4().hex
        parts = [
            f'--{boundary}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'.encode(),
            (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
             f"Content-Type: video/mp4\r\n\r\n").encode(),
            Path(file_path).read_bytes(),
            f"\r\n--{boundary}--\r\n".encode(),
        ]
        res = self.request("/upload/image", b"".join(parts),
                           {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        return f"{res['subfolder']}/{res['name']}" if res.get("subfolder") else res["name"]

    def preflight(self, models, need_rife, need_windows):
        stats = self.request("/system_stats")
        dev = (stats.get("devices") or [{}])[0]
        log(f"ComfyUI {stats['system'].get('comfyui_version', '?')} on {dev.get('name', '?')}")
        nodes = ["WanVaceToVideo", "TrimVideoLatent", "LoadVideo", "GetVideoComponents", "SaveWEBM"]
        if need_windows:
            nodes.append("WanContextWindowsManual")
        if need_rife:
            nodes += ["FrameInterpolationModelLoader", "FrameInterpolate", "CreateVideo", "VideoFrameSample"]
        info = self.request("/object_info")
        missing = [n for n in nodes if n not in info]
        if missing:
            die(f"your ComfyUI is missing core nodes {missing}; update ComfyUI.")
        folders = {"unet": "diffusion_models", "clip": "text_encoders", "vae": "vae"}
        if need_rife:
            folders["rife"] = "frame_interpolation"
        for key, folder in folders.items():
            available = self.request(f"/models/{folder}")
            if models[key] not in available:
                die(f"model '{models[key]}' not found in models/{folder}/ (have: {available}). See README.")
        free = dev.get("vram_free", 0) / 2**20
        if free and free < 4500:
            log(f"WARNING: only {free:.0f} MB VRAM free. Close other GPU apps if it runs out of memory.")

    def run(self, workflow):
        client_id = uuid.uuid4().hex
        prompt_id = self.post_json("/prompt", {"prompt": workflow, "client_id": client_id})["prompt_id"]
        log(f"queued {prompt_id}")
        start, last = time.time(), 0
        try:
            while True:
                hist = self.request(f"/history/{prompt_id}").get(prompt_id)
                if hist and hist.get("status", {}).get("completed") is not None:
                    break
                if time.time() - last > 60:
                    last = time.time()
                    log(f"  generating... {(last - start) / 60:.0f} min")
                time.sleep(5)
        except KeyboardInterrupt:
            self.request("/interrupt", b"")
            die("interrupted; ComfyUI job cancelled.")
        status = hist["status"]
        if status.get("status_str") != "success":
            for kind, msg in status.get("messages", []):
                if kind == "execution_error":
                    print(json.dumps(msg, indent=2)[:4000], file=sys.stderr)
            die(f"ComfyUI job failed ({status.get('status_str')}).")
        log(f"done in {(time.time() - start) / 60:.1f} min")
        return hist["outputs"], time.time() - start

    def download(self, item, dest):
        q = urllib.parse.urlencode({"filename": item["filename"], "subfolder": item.get("subfolder", ""),
                                    "type": item.get("type", "output")})
        Path(dest).write_bytes(self.request(f"/view?{q}", raw=True, timeout=300))


# ---------------------------------------------------------------- video I/O

def frames_to_mp4(frames_dir, out_path, fps):
    import av
    import numpy as np
    from PIL import Image

    files = sorted(Path(frames_dir).glob("*.png"))
    if not files:
        die(f"no frames in {frames_dir}")
    with av.open(str(out_path), "w") as container:
        w, h = Image.open(files[0]).size
        stream = container.add_stream("libx264", rate=Fraction(fps))
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        stream.options = {"crf": "12"}
        for f in files:
            img = np.asarray(Image.open(f).convert("RGB"))
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return len(files)


def read_frames(path):
    import av
    with av.open(str(path)) as c:
        return [f.to_image() for f in c.decode(c.streams.video[0])]


def write_mp4(images, out_path, fps):
    import av
    with av.open(str(out_path), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(fps))
        stream.width, stream.height = images[0].size
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "16"}
        for img in images:
            for packet in stream.encode(av.VideoFrame.from_image(img.convert("RGB"))):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def side_by_side(gray_dir, result_frames, out_path, fps):
    from PIL import Image
    gray = sorted(Path(gray_dir).glob("*.png"))
    if not gray:
        return
    w, h = result_frames[0].size
    combined = []
    for k, frame in enumerate(result_frames):
        g = Image.open(gray[min(len(gray) - 1, round(k * WAN_FPS / fps))]).convert("RGB").resize((w, h))
        canvas = Image.new("RGB", (w * 2, h))
        canvas.paste(g, (0, 0))
        canvas.paste(frame, (w, 0))
        combined.append(canvas)
    write_mp4(combined, out_path, fps)


# ---------------------------------------------------------------- workflow

def build_workflow(a, frames, depth_file, prefix):
    m = a.models
    wf = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": m["unet"], "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["clip"], "type": "wan", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": m["vae"]}},
        "shift": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["unet", 0], "shift": a.shift}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": a.prompt}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": a.negative}},
        "load": {"class_type": "LoadVideo", "inputs": {"file": depth_file}},
        "frames": {"class_type": "GetVideoComponents", "inputs": {"video": ["load", 0]}},
        "vace": {"class_type": "WanVaceToVideo", "inputs": {
            "positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0],
            "width": a.width, "height": a.height, "length": frames, "batch_size": 1,
            "strength": a.strength, "control_video": ["frames", 0]}},
        "sample": {"class_type": "KSampler", "inputs": {
            "model": ["shift", 0], "positive": ["vace", 0], "negative": ["vace", 1],
            "latent_image": ["vace", 2], "seed": a.seed, "steps": a.steps, "cfg": a.cfg,
            "sampler_name": "uni_pc", "scheduler": "simple", "denoise": 1.0}},
        "trim": {"class_type": "TrimVideoLatent", "inputs": {"samples": ["sample", 0], "trim_amount": ["vace", 3]}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["trim", 0], "vae": ["vae", 0]}},
        "save16": {"class_type": "SaveWEBM", "inputs": {
            "images": ["decode", 0], "filename_prefix": f"{prefix}_16fps", "codec": "vp9",
            "fps": float(WAN_FPS), "crf": 16.0}},
    }
    if frames > WINDOW:
        wf["windows"] = {"class_type": "WanContextWindowsManual", "inputs": {
            "model": ["shift", 0], "context_length": WINDOW, "context_overlap": a.context_overlap,
            "context_schedule": "standard_uniform", "context_stride": 1, "closed_loop": False,
            "fuse_method": "pyramid", "freenoise": True, "retain_first_frame": False,
            "split_conds_to_windows": False}}
        wf["sample"]["inputs"]["model"] = ["windows", 0]
    if a.fps == 24:
        # RIFE x3 -> 48 fps, keep every 2nd frame -> 24 fps (exact for 4n+1 frame counts)
        wf.update({
            "rife": {"class_type": "FrameInterpolationModelLoader", "inputs": {"model_name": m["rife"]}},
            "interp": {"class_type": "FrameInterpolate", "inputs": {
                "interp_model": ["rife", 0], "images": ["decode", 0], "multiplier": 3}},
            "vid48": {"class_type": "CreateVideo", "inputs": {"images": ["interp", 0], "fps": 48.0}},
            "pick": {"class_type": "VideoFrameSample", "inputs": {
                "video": ["vid48", 0], "num_frames": 3 * (frames - 1) // 2 + 1,
                "strategy": "uniform", "seed": 0}},
            "frames24": {"class_type": "GetVideoComponents", "inputs": {"video": ["pick", 0]}},
            "save24": {"class_type": "SaveWEBM", "inputs": {
                "images": ["frames24", 0], "filename_prefix": f"{prefix}_24fps", "codec": "vp9",
                "fps": 24.0, "crf": 16.0}},
        })
    return wf


def saved_files(outputs, node_id):
    items = []
    for value in outputs.get(node_id, {}).values():
        if isinstance(value, list):
            items += [i for i in value if isinstance(i, dict) and "filename" in i]
    return items


# ---------------------------------------------------------------- main

def interactive(a, parser):
    """Ask for the essentials when started without a prompt (e.g. double-clicked)."""
    def ask(question, default):
        try:
            answer = input(f"{question} [{default}]: ").strip().strip('"')
        except EOFError:
            answer = ""
        return answer or default

    print("\n=== blender-wan-6gb ===")
    if a.blend:
        print(f"Scene: {a.blend}")
    else:
        print("Scene: built-in test scene (drag a .blend onto make_video.bat to use your own)")
    while not a.prompt:
        try:
            a.prompt = input("Describe the video (English works best): ").strip()
        except EOFError:
            parser.error("--prompt is required")
    try:
        a.seconds = float(ask("Length in seconds", f"{a.seconds:g}"))
        a.fps = int(ask("Frames per second, 16 or 24", str(a.fps)))
    except ValueError:
        parser.error("seconds and fps must be numbers")
    if a.fps not in (16, 24):
        parser.error("fps must be 16 or 24")
    frames = wan_frame_count(a.seconds)
    windows = math.ceil((frames - WINDOW) / (WINDOW - a.context_overlap)) + 1 if frames > WINDOW else 1
    minutes = 15 * windows if frames >= WINDOW else max(3, round(15 * frames / WINDOW))
    print(f"Roughly {minutes} min on a 6 GB laptop GPU at 832x480. Press Ctrl+C to cancel.\n")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("blend_file", nargs="?", help="same as --blend (lets you drag a .blend onto make_video.bat)")
    src = p.add_argument_group("scene")
    src.add_argument("--blend", help=".blend file with an animated camera (omit to use the built-in test scene)")
    src.add_argument("--camera", help="camera object name (default: the scene camera)")
    src.add_argument("--start", type=float, help="scene frame to start from (default: scene start frame)")
    src.add_argument("--near", type=float, help="depth mapped to white, in scene units (default: auto)")
    src.add_argument("--far", type=float, help="depth mapped to black, in scene units (default: auto)")
    src.add_argument("--blender", help="path to blender executable (default: auto-detect)")

    gen = p.add_argument_group("generation")
    gen.add_argument("--prompt", help="what the scene should look like (required unless --depth-only)")
    gen.add_argument("--negative", default=DEFAULT_NEGATIVE)
    gen.add_argument("--seconds", type=float, default=5.0, help="clip length (default 5; >5 s uses context windows)")
    gen.add_argument("--fps", type=int, choices=[16, 24], default=24, help="output fps (24 = RIFE interpolation)")
    gen.add_argument("--width", type=int, default=832)
    gen.add_argument("--height", type=int, default=480)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--steps", type=int, default=20)
    gen.add_argument("--cfg", type=float, default=6.0)
    gen.add_argument("--shift", type=float, default=8.0)
    gen.add_argument("--strength", type=float, default=1.0, help="how strictly to follow the depth (VACE strength)")
    gen.add_argument("--context-overlap", type=int, default=30, help="frames shared between context windows")

    run = p.add_argument_group("run")
    run.add_argument("--name", help="output folder name (default: timestamp)")
    run.add_argument("--out-dir", default=str(HERE / "outputs"))
    run.add_argument("--server", default=os.environ.get("COMFY_SERVER") or CONFIG.get("server")
                     or "http://127.0.0.1:8188")
    run.add_argument("--depth-only", action="store_true", help="only render the Blender depth/preview, no ComfyUI")
    run.add_argument("--reuse-depth", action="store_true", help="skip Blender if this --name already has depth frames")
    run.add_argument("--no-preview", action="store_true", help="skip the gray preview / side-by-side video")
    for key, default in DEFAULT_MODELS.items():
        run.add_argument(f"--{key}-model", default=default, help=argparse.SUPPRESS)
    a = p.parse_args()
    a.blend = a.blend or a.blend_file
    if not a.depth_only and not a.prompt:
        if not sys.stdin.isatty():
            p.error("--prompt is required (or use --depth-only)")
        interactive(a, p)
    if a.width % 16 or a.height % 16:
        p.error("--width and --height must be multiples of 16")
    a.models = {k: getattr(a, f"{k}_model") for k in DEFAULT_MODELS}
    return a


def main():
    a = parse_args()
    frames = wan_frame_count(a.seconds)
    name = a.name or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not re.fullmatch(r"[\w.-]+", name):
        die("--name may only contain letters, digits, '_', '-' and '.'")
    out = Path(a.out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    windows = math.ceil((frames - WINDOW) / (WINDOW - a.context_overlap)) + 1 if frames > WINDOW else 1
    log(f"{frames} frames at 16 fps ({(frames - 1) / WAN_FPS:.2f} s), {windows} window(s), output {a.fps} fps -> {out}")

    comfy = None
    if not a.depth_only:  # fail fast before the Blender render
        comfy = Comfy(a.server)
        comfy.preflight(a.models, need_rife=a.fps == 24, need_windows=frames > WINDOW)

    # 1. depth from Blender
    depth_dir = out / "render" / "depth"
    if a.reuse_depth and len(list(depth_dir.glob("*.png"))) == frames:
        log("reusing existing depth frames")
    else:
        blender = find_blender(a.blender)
        cmd = [blender, "-b"]
        if a.blend:
            if not Path(a.blend).is_file():
                die(f"{a.blend} not found")
            cmd.append(str(Path(a.blend).resolve()))
        cmd += ["--factory-startup", "-P", str(HERE / "blender" / "render_depth.py"), "--",
                "--out", str(out / "render"), "--frames", str(frames),
                "--width", str(a.width), "--height", str(a.height)]
        if not a.blend:
            cmd.append("--test-scene")
        for flag in ("camera", "start", "near", "far"):
            if getattr(a, flag) is not None:
                cmd += [f"--{flag}", str(getattr(a, flag))]
        if a.no_preview:
            cmd.append("--no-preview")
        shutil.rmtree(out / "render", ignore_errors=True)
        log(f"rendering depth in Blender ({Path(blender).name})...")
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        done = [l for l in proc.stdout.splitlines() if l.startswith("DEPTH_DONE")]
        if proc.returncode or not done:
            print(proc.stdout[-3000:], proc.stderr[-3000:], file=sys.stderr)
            die("Blender render failed")
        info = json.loads(done[0].split(" ", 1)[1])
        log(f"depth range near={info['near']:.2f} far={info['far']:.2f} (override with --near/--far)")
    frames_to_mp4(depth_dir, out / "depth.mp4", WAN_FPS)
    if (out / "render" / "gray").is_dir():
        frames_to_mp4(out / "render" / "gray", out / "preview_gray.mp4", WAN_FPS)
    if a.depth_only:
        log(f"depth-only: check {out / 'depth.mp4'} and {out / 'preview_gray.mp4'}")
        return

    # 2. generate in ComfyUI
    depth_file = comfy.upload(out / "depth.mp4", f"wanblender_{name}_depth.mp4")
    workflow = build_workflow(a, frames, depth_file, f"wan_blender/{name}")
    (out / "workflow_api.json").write_text(json.dumps(workflow, indent=1))
    outputs, seconds = comfy.run(workflow)

    # 3. download + convert to mp4
    final_node = "save24" if a.fps == 24 else "save16"
    for node, label in (("save16", "result_16fps"), ("save24", "result_24fps")):
        for item in saved_files(outputs, node):
            comfy.download(item, out / f"{label}.webm")
    final_webm = out / f"result_{a.fps}fps.webm"
    if not final_webm.exists():
        die(f"no output from node {final_node}")
    result = read_frames(final_webm)
    write_mp4(result, out / "result.mp4", a.fps)
    if not a.no_preview and (out / "render" / "gray").is_dir():
        side_by_side(out / "render" / "gray", result, out / "compare.mp4", a.fps)

    settings = {k: v for k, v in vars(a).items() if not k.endswith("_model")}
    settings.update(wan_frames=frames, windows=windows, generation_seconds=round(seconds))
    (out / "settings.json").write_text(json.dumps(settings, indent=2))
    log(f"result: {out / 'result.mp4'} ({len(result)} frames, {len(result) / a.fps:.2f} s @ {a.fps} fps)")


if __name__ == "__main__":
    main()
