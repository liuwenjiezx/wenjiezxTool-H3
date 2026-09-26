"""Backend for the MiniMax H3 Timeline Director node.

The UI stores an edit decision list (EDL) in ``timeline_data``.  At queue time
this module resolves the current generation selection into H3 references and
native ``MiniMaxH3AddGuide`` keyframe guides:

* only the portions overlapping the generated range become reference videos;
* those same overlapping portions may become fixed guide clips;
* a gap between two clips uses native first/last single-frame guides;
* standalone image and audio assets are forwarded as references.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib
import inspect
import json
import logging
import math
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable

import av
import numpy as np
import torch
from aiohttp import web
from PIL import Image, ImageOps

import folder_paths
import node_helpers
from comfy_api.input_impl import VideoFromFile
from comfy_api.input import VideoInput
from comfy_api.latest import io
from comfy_extras import nodes_minimax_h3 as h3_nodes
from server import PromptServer


log = logging.getLogger("WJZ_H3_TimelineDirector")
FPS = 24.0
MAX_REF_IMAGES = 9
MAX_REF_VIDEOS = 3
MAX_REF_AUDIOS = 3
MAX_REF_VIDEO_SECONDS = 15.0
MIN_REF_VIDEO_SECONDS = 5.0 / FPS
UPLOAD_SUBDIR = "minimax_h3_timeline_director"
PREVIEW_SUBDIR = f"{UPLOAD_SUBDIR}/preview_proxies"
PREVIEW_MAX_WIDTH = 480
PREVIEW_MAX_HEIGHT = 270
PREVIEW_FPS = 12
MAX_HTTP_MEDIA_BYTES = 512 * 1024 * 1024
MAX_HTTP_MEDIA_SECONDS = 30 * 60.0
MAX_PREVIEW_SOURCE_SECONDS = 15 * 60.0
MAX_HTTP_MEDIA_DIMENSION = 16384
MAX_HTTP_MEDIA_PIXELS = 8192 * 8192
MAX_PREVIEW_CACHE_BYTES = 2 * 1024 * 1024 * 1024
MAX_PREVIEW_CACHE_FILES = 256
MEDIA_INFO_CONCURRENCY = 2
PREVIEW_CONCURRENCY = 1
MEDIA_EXTENSIONS = {
    "image": {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"},
    "audio": {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma"},
    "video": {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpeg", ".mpg"},
}
_MEDIA_INFO_SEMAPHORE = asyncio.Semaphore(MEDIA_INFO_CONCURRENCY)
_PREVIEW_SEMAPHORE = asyncio.Semaphore(PREVIEW_CONCURRENCY)

TimelinePlan = io.Custom("MINIMAX_H3_TIMELINE_PLAN")
PromptMediaBundle = io.Custom("MINIMAX_H3_OMNI_MEDIA_BUNDLE")
PROMPT_REWRITER_OPTIONS = io.Custom("H3_REWRITER_OPTIONS")
OMNI_REWRITER_DIRECTORY = "MiniMax-H3-Prompt-Rewriter-ComfyUI"
OMNI_MISSING_MODEL = "Install MiniMax-H3-Prompt-Rewriter-ComfyUI first"
# The bridge node is verified against this upstream version. Major/minor mismatches
# only emit a warning; arrange/rewrite_omni failures become readable compatibility errors.
OMNI_ADAPTED_VERSION = "0.17"


class TrimmedTimelineVideo(VideoInput):
    """A standard VIDEO that cannot leak its untrimmed source to consumers.

    MiniMax-H3 Prompt Rewriter Omni intentionally reaches into a plain
    ``VideoFromFile`` to sample the original path quickly.  That optimization
    bypasses ``start_time`` and ``duration``.  This wrapper exposes only the
    standard ``VideoInput`` surface, so Omni falls back to ``save_to`` and sees
    exactly the same trimmed interval later encoded as ``ref_video_N``.
    """

    def __init__(self, path: str, start_time: float, duration: float):
        self._path = str(path)
        self._start_time = max(0.0, float(start_time))
        self._duration = max(0.0, float(duration))
        self._delegate = VideoFromFile(
            self._path, start_time=self._start_time, duration=self._duration
        )

    def get_components(self):
        return self._delegate.get_components()

    def save_to(self, path, format=None, codec=None, metadata=None, bit_depth=None, crf=None, color_space=None):
        kwargs = {
            "metadata": metadata,
            "bit_depth": bit_depth,
            "crf": crf,
            "color_space": color_space,
        }
        if format is not None:
            kwargs["format"] = format
        if codec is not None:
            kwargs["codec"] = codec
        return self._delegate.save_to(path, **kwargs)

    def as_trimmed(self, start_time=None, duration=None, strict_duration=False):
        relative_start = max(0.0, float(start_time or 0.0))
        if relative_start >= self._duration:
            return None
        available = self._duration - relative_start
        requested = available if duration is None or float(duration) <= 0 else float(duration)
        if strict_duration and requested > available:
            return None
        return TrimmedTimelineVideo(
            self._path, self._start_time + relative_start, min(requested, available)
        )

    def get_active_trim_window(self) -> tuple[float, float]:
        return self._start_time, self._duration

    def get_dimensions(self):
        return self._delegate.get_dimensions()

    def get_bit_depth(self):
        return self._delegate.get_bit_depth()

    def get_color_space(self):
        return self._delegate.get_color_space()

    def get_duration(self):
        return self._duration

    def get_frame_count(self):
        return self._delegate.get_frame_count()

    def get_frame_rate(self):
        return self._delegate.get_frame_rate()

    def get_container_format(self):
        return self._delegate.get_container_format()


def _load_omni_rewriter():
    """加载可选的提示词改写器插件，不将其设为必需依赖。"""

    try:
        return importlib.import_module("minimax_h3_rewriter.writer_omni")
    except ModuleNotFoundError as first_error:
        for custom_nodes_root in folder_paths.get_folder_paths("custom_nodes"):
            candidate = Path(custom_nodes_root) / OMNI_REWRITER_DIRECTORY
            if candidate.is_dir() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
                try:
                    return importlib.import_module("minimax_h3_rewriter.writer_omni")
                except ModuleNotFoundError:
                    continue
        raise RuntimeError(
            "未找到 MiniMax-H3-Prompt-Rewriter-ComfyUI；使用 H3 全媒体提示词桥接前请先安装它。"
        ) from first_error


def _omni_version() -> str:
    """尽力读取已安装改写器插件的版本号（未知时返回空字符串）。"""

    try:
        module = _load_omni_rewriter()
        root = Path(module.__file__).resolve().parents[1]
        pyproject = root / "pyproject.toml"
        if pyproject.is_file():
            import tomllib

            with pyproject.open("rb") as handle:
                data = tomllib.load(handle)
            project = data.get("project") or data.get("tool", {}).get("poetry", {})
            version = project.get("version")
            if version:
                return str(version)
    except Exception:
        pass
    return ""


def _omni_compatibility_warning() -> str | None:
    """当已安装的改写器版本与本桥接器不匹配时给出提示。"""

    installed = _omni_version()
    if not installed:
        return None
    if tuple(installed.split(".")[:2]) == tuple(OMNI_ADAPTED_VERSION.split(".")):
        return None
    return (
        f"检测到 MiniMax-H3-Prompt-Rewriter-ComfyUI v{installed}；"
        f"本全媒体提示词桥接器面向的是 v{OMNI_ADAPTED_VERSION}。若执行失败，请把改写器固定在 v0.17.x。"
    )


def _omni_interface_error(operation: str, exc: Exception) -> RuntimeError:
    installed = _omni_version() or "unknown"
    return RuntimeError(
        f"MiniMax-H3-Prompt-Rewriter-ComfyUI v{installed} 与全媒体提示词桥接器不兼容"
        f"（本桥接器面向 v{OMNI_ADAPTED_VERSION}）：{operation} 失败：{exc}。\n"
        "Pin the rewriter to v0.17.x or update this plugin and try again."
    )


def _call_omni_rewrite(module, **kwargs):
    """Call rewrite_omni with only the keyword arguments the installed version accepts.

    Newer versions may rename or drop parameters; passing only the parameters
    present in the live signature keeps a minor upstream drift from crashing.
    """

    signature = inspect.signature(module.rewrite_omni)
    accepted = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    filtered = {name: value for name, value in kwargs.items() if name in accepted}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        log.warning("Omni rewrite_omni 忽略了未声明的参数：%s", ", ".join(dropped))
    return module.rewrite_omni(**filtered)


def _omni_schema_choices() -> tuple[list[str], list[str], list[str]]:
    try:
        module = _load_omni_rewriter()
        return (
            [str(value).upper() for value in module.TASKS],
            [str(value) for value in module.model_choices()],
            [str(value) for value in module.QUANTIZATIONS],
        )
    except Exception:
        return ["REF2AV"], [OMNI_MISSING_MODEL], ["nf4", "int8", "bfloat16", "float16"]


def _closest_omni_resolution(width: int, height: int) -> str:
    choices = {
        "21:9": 21 / 9,
        "16:9": 16 / 9,
        "4:3": 4 / 3,
        "1:1": 1.0,
        "3:4": 3 / 4,
        "9:16": 9 / 16,
    }
    ratio = max(1, int(width)) / max(1, int(height))
    return min(choices, key=lambda name: abs(math.log(ratio / choices[name])))


def _upload_root() -> Path:
    root = Path(folder_paths.get_input_directory()).resolve() / UPLOAD_SUBDIR
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_name(value: str) -> str:
    name = Path(str(value or "media")).name
    stem = re.sub(r"[^\w.()\-\u4e00-\u9fff]+", "_", name, flags=re.UNICODE)
    return (stem or "media")[:180]


def _safe_input_path(relative_name: str) -> Path:
    root = Path(folder_paths.get_input_directory()).resolve()
    candidate = (root / str(relative_name or "").replace("/", os.sep)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("媒体路径不在 ComfyUI 输入目录内") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"找不到参考媒体：{relative_name}")
    return candidate


def _safe_uploaded_path(relative_name: str, kind: str | None = None) -> Path:
    """Resolve a browser-facing media path inside this plugin's upload area.

    Node execution intentionally continues to accept existing files anywhere
    below ComfyUI's input directory.  HTTP helpers are narrower: a remote
    caller may only inspect files uploaded through this plugin's fixed
    subfolder, and may never feed a generated preview back into the helpers.
    """

    candidate = _safe_input_path(relative_name)
    upload_root = _upload_root()
    try:
        relative = candidate.relative_to(upload_root)
    except ValueError as exc:
        raise ValueError("媒体不在导演台的上传文件夹里") from exc
    if len(relative.parts) != 1:
        raise ValueError("只有直接上传的文件才能检查或预览")
    if kind is not None and kind not in MEDIA_EXTENSIONS:
        raise ValueError("不支持的媒体类型")
    if kind and candidate.suffix.lower() not in MEDIA_EXTENSIONS[kind]:
        raise ValueError(f"不支持的媒体类型：{kind}（文件扩展名不被支持）")
    return candidate


def _relative_input(path: Path) -> str:
    root = Path(folder_paths.get_input_directory()).resolve()
    return path.resolve().relative_to(root).as_posix()


def _preview_root() -> Path:
    root = Path(folder_paths.get_input_directory()).resolve() / Path(PREVIEW_SUBDIR)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _validate_http_media_file(path: Path) -> None:
    size = path.stat().st_size
    if size <= 0:
        raise ValueError("媒体文件为空")
    if size > MAX_HTTP_MEDIA_BYTES:
        raise ValueError("媒体文件超过 512 MiB 安全上限")


def _validate_http_media_info(
    info: dict[str, Any], kind: str, *, max_duration: float = MAX_HTTP_MEDIA_SECONDS
) -> None:
    duration = max(0.0, _float(info.get("duration")))
    width = max(0, int(_float(info.get("width"))))
    height = max(0, int(_float(info.get("height"))))
    if kind in {"audio", "video"} and duration <= 0:
        raise ValueError("无法安全判定媒体时长")
    if duration > max_duration:
        raise ValueError(f"媒体时长超过 {int(max_duration // 60)} 分钟安全上限")
    if width > MAX_HTTP_MEDIA_DIMENSION or height > MAX_HTTP_MEDIA_DIMENSION:
        raise ValueError("媒体尺寸超过安全上限")
    if width and height and width * height > MAX_HTTP_MEDIA_PIXELS:
        raise ValueError("媒体像素总数超过安全上限")
    if kind == "video" and not info.get("hasVideo"):
        raise ValueError("上传的文件没有视频轨")
    if kind == "audio" and not info.get("hasAudio"):
        raise ValueError("上传的文件没有音频轨")


def _uploaded_media_info(path: Path, kind: str) -> dict[str, Any]:
    """Validate one official ComfyUI upload and return bounded UI metadata."""

    _validate_http_media_file(path)
    if kind == "image":
        with Image.open(path) as image:
            width, height = image.size
            info = {
                "filename": _relative_input(path),
                "name": path.name,
                "duration": 0.0,
                "width": int(width),
                "height": int(height),
                "fps": 0.0,
                "hasVideo": False,
                "hasAudio": False,
                "peaks": [],
            }
            _validate_http_media_info(info, kind)
            image.verify()
        return info

    info = _media_info(path, include_peaks=False)
    _validate_http_media_info(info, kind)
    if info["hasAudio"]:
        info["peaks"] = _audio_peaks(path)
    return info


def _prune_preview_cache(keep: Path) -> None:
    """Bound the disk used by generated monitoring proxies."""

    files = sorted(
        (item for item in _preview_root().glob("*.mp4") if item.is_file()),
        key=lambda item: item.stat().st_mtime_ns,
        reverse=True,
    )
    total = sum(item.stat().st_size for item in files)
    for index, item in enumerate(files):
        if item == keep:
            continue
        if index < MAX_PREVIEW_CACHE_FILES and total <= MAX_PREVIEW_CACHE_BYTES:
            continue
        size = item.stat().st_size
        item.unlink(missing_ok=True)
        total = max(0, total - size)


def _ensure_preview_proxy(source: Path) -> str:
    """Create a cached, silent low-resolution MP4 for timeline monitoring."""

    _validate_http_media_file(source)
    source_info = _media_info(source, include_peaks=False)
    _validate_http_media_info(source_info, "video", max_duration=MAX_PREVIEW_SOURCE_SECONDS)

    stat = source.stat()
    fingerprint = hashlib.sha1(
        f"{source.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|v1".encode("utf-8")
    ).hexdigest()[:16]
    destination = _preview_root() / f"{_safe_name(source.stem)[:80]}_{fingerprint}.mp4"
    if destination.is_file() and destination.stat().st_size > 1024:
        return _relative_input(destination)

    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("当前 ComfyUI 环境未提供 ffmpeg，无法生成低清预览") from exc

    temporary = destination.with_name(f".{destination.stem}_{uuid.uuid4().hex}.tmp.mp4")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-map", "0:v:0", "-an", "-vf",
        (
            f"fps={PREVIEW_FPS},scale={PREVIEW_MAX_WIDTH}:{PREVIEW_MAX_HEIGHT}:"
            "force_original_aspect_ratio=decrease:force_divisible_by=2"
        ),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
    ]
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False,
            creationflags=creation_flags, timeout=180,
        )
        if completed.returncode != 0 or not temporary.is_file():
            detail = (completed.stderr or completed.stdout or "unknown ffmpeg error").strip()[-1200:]
            raise RuntimeError(f"低清预览生成失败：{detail}")
        os.replace(temporary, destination)
        _prune_preview_cache(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return _relative_input(destination)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _media_info(path: Path, include_peaks: bool = True) -> dict[str, Any]:
    info: dict[str, Any] = {
        "filename": _relative_input(path),
        "name": path.name,
        "duration": 0.0,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "hasVideo": False,
        "hasAudio": False,
        "peaks": [],
    }
    with av.open(str(path)) as container:
        if container.duration is not None:
            info["duration"] = max(0.0, float(container.duration / av.time_base))
        if container.streams.video:
            stream = container.streams.video[0]
            info.update(
                hasVideo=True,
                width=int(stream.width or stream.codec_context.width or 0),
                height=int(stream.height or stream.codec_context.height or 0),
                fps=float(stream.average_rate or stream.guessed_rate or 0),
            )
            if not info["duration"] and stream.duration is not None and stream.time_base:
                info["duration"] = float(stream.duration * stream.time_base)
        if container.streams.audio:
            info["hasAudio"] = True
            stream = container.streams.audio[0]
            if not info["duration"] and stream.duration is not None and stream.time_base:
                info["duration"] = float(stream.duration * stream.time_base)
    if include_peaks and info["hasAudio"]:
        try:
            info["peaks"] = _audio_peaks(path)
        except Exception as exc:
            log.debug("Waveform extraction failed for %s: %s", path, exc)
    return info


def _decode_audio(path: Path, start: float = 0.0, duration: float | None = None) -> dict[str, Any] | None:
    """Decode an audio interval to ComfyUI AUDIO at 44.1 kHz.

    Seeking starts slightly early so compressed sources do not lose samples at
    a keyframe/packet boundary.  The decoded waveform is then sliced using the
    first decoded timestamp.
    """

    target_rate = 44100
    start = max(0.0, float(start))
    end = None if duration is None else start + max(0.0, float(duration))
    chunks: list[np.ndarray] = []
    first_time: float | None = None
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        seek_to = max(0.0, start - 1.0)
        if seek_to > 0:
            container.seek(int(seek_to * av.time_base), backward=True)
        layout = "stereo" if getattr(stream.codec_context, "channels", 1) > 1 else "mono"
        resampler = av.AudioResampler(format="fltp", layout=layout, rate=target_rate)
        stop_after = None if end is None else end + 1.0
        for frame in container.decode(stream):
            frame_time = float(frame.time) if frame.time is not None else seek_to
            if first_time is None:
                first_time = frame_time
            if stop_after is not None and frame_time > stop_after:
                break
            converted = resampler.resample(frame)
            if not isinstance(converted, list):
                converted = [converted]
            for out in converted:
                arr = out.to_ndarray()
                if arr.ndim == 1:
                    arr = arr[None, :]
                chunks.append(arr.astype(np.float32, copy=False))
        flushed = resampler.resample(None)
        if not isinstance(flushed, list):
            flushed = [flushed] if flushed is not None else []
        for out in flushed:
            arr = out.to_ndarray()
            if arr.ndim == 1:
                arr = arr[None, :]
            chunks.append(arr.astype(np.float32, copy=False))
    if not chunks:
        return None
    waveform = np.concatenate(chunks, axis=1)
    decoded_start = max(0.0, first_time or 0.0)
    begin = max(0, int(round((start - decoded_start) * target_rate)))
    finish = waveform.shape[1] if end is None else min(
        waveform.shape[1], int(round((end - decoded_start) * target_rate))
    )
    if finish <= begin:
        return None
    tensor = torch.from_numpy(np.ascontiguousarray(waveform[:, begin:finish])).unsqueeze(0)
    return {"waveform": tensor, "sample_rate": target_rate}


def _audio_mode(asset: dict[str, Any]) -> str:
    """Normalize persisted audio-card behavior without breaking older workflows."""

    return "locked" if str(asset.get("audioMode") or "").lower() == "locked" else "reference"


def _locked_audio_assets(timeline: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        asset for asset in timeline.get("audios", [])
        if isinstance(asset, dict) and asset.get("file") and _audio_mode(asset) == "locked"
    ]


def _empty_audio(sample_rate: int = 44100) -> dict[str, Any]:
    return {"waveform": torch.zeros((1, 1, 1), dtype=torch.float32), "sample_rate": sample_rate}


def _audio_channels(waveform: torch.Tensor, channels: int) -> torch.Tensor:
    """Normalize ComfyUI [B,C,T] audio to a common channel count."""

    waveform = waveform[:1].to(dtype=torch.float32, device="cpu")
    current = waveform.shape[1]
    if current == channels:
        return waveform
    if current == 1:
        return waveform.repeat(1, channels, 1)
    if channels == 1:
        return waveform.mean(dim=1, keepdim=True)
    if current > channels:
        return waveform[:, :channels]
    return torch.cat([waveform, waveform[:, -1:].repeat(1, channels - current, 1)], dim=1)


def _timeline_video_audio(timeline: dict[str, Any], sample_rate: int = 44100) -> dict[str, Any]:
    """Mix every edited video soundtrack at its timeline position, including gaps."""

    decoded: list[tuple[int, torch.Tensor]] = []
    channels = 1
    timeline_samples = 0
    clips = [c for c in timeline.get("videoClips", []) if isinstance(c, dict) and c.get("file")]
    for clip in sorted(clips, key=lambda item: _float(item.get("start"))):
        start = max(0.0, _float(clip.get("start")))
        duration = max(0.0, _float(clip.get("duration")))
        timeline_samples = max(timeline_samples, int(round((start + duration) * sample_rate)))
        if not clip.get("hasAudio", True) or duration <= 0:
            continue
        audio = _decode_audio(
            _safe_input_path(str(clip["file"])), max(0.0, _float(clip.get("trimStart"))), duration
        )
        if audio is None:
            continue
        waveform = audio["waveform"]
        channels = max(channels, int(waveform.shape[1]))
        decoded.append((int(round(start * sample_rate)), waveform))
    if not decoded:
        return {
            "waveform": torch.zeros((1, 1, max(1, timeline_samples)), dtype=torch.float32),
            "sample_rate": sample_rate,
        }
    mixed = torch.zeros((1, channels, max(1, timeline_samples)), dtype=torch.float32)
    for offset, waveform in decoded:
        normalized = _audio_channels(waveform, channels)
        available = max(0, mixed.shape[-1] - offset)
        count = min(available, normalized.shape[-1])
        if count:
            mixed[..., offset : offset + count] += normalized[..., :count]
    return {"waveform": mixed.clamp_(-1.0, 1.0), "sample_rate": sample_rate}


def _standalone_audio_track(timeline: dict[str, Any], sample_rate: int = 44100) -> dict[str, Any]:
    """Concatenate independent reference audios in their visible upload order."""

    waveforms: list[torch.Tensor] = []
    channels = 1
    for asset in timeline.get("audios", []):
        if not isinstance(asset, dict) or not asset.get("file"):
            continue
        duration = _float(asset.get("duration")) or None
        audio = _decode_audio(
            _safe_input_path(str(asset["file"])), max(0.0, _float(asset.get("trimStart"))), duration
        )
        if audio is None:
            continue
        waveform = audio["waveform"]
        channels = max(channels, int(waveform.shape[1]))
        waveforms.append(waveform)
    if not waveforms:
        return _empty_audio(sample_rate)
    merged = torch.cat([_audio_channels(waveform, channels) for waveform in waveforms], dim=-1)
    return {"waveform": merged, "sample_rate": sample_rate}


def _audio_peaks(path: Path, count: int = 180) -> list[float]:
    """Build a fixed-size waveform preview without retaining the full soundtrack."""

    preview_rate = 8000
    peaks = np.zeros(count, dtype=np.float32)
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return []
        stream = container.streams.audio[0]
        duration = float(container.duration / av.time_base) if container.duration else 0.0
        if not duration and stream.duration is not None and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        # Unknown-duration streams are rare for uploaded files.  Give them a
        # conservative ten-minute preview span while keeping memory constant.
        total_samples = max(count, int(math.ceil(max(duration, 600.0 if not duration else duration) * preview_rate)))
        bin_size = max(1, int(math.ceil(total_samples / count)))
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=preview_rate)
        position = 0

        def consume(array: np.ndarray) -> None:
            nonlocal position
            values = np.abs(array.reshape(-1).astype(np.float32, copy=False))
            offset = 0
            while offset < values.size:
                bucket = min(count - 1, position // bin_size)
                take = min(values.size - offset, bin_size - (position % bin_size))
                if take > 0:
                    peaks[bucket] = max(peaks[bucket], float(values[offset : offset + take].max(initial=0.0)))
                position += take
                offset += take

        for frame in container.decode(stream):
            converted = resampler.resample(frame)
            if not isinstance(converted, list):
                converted = [converted]
            for out in converted:
                consume(out.to_ndarray())
        flushed = resampler.resample(None)
        if not isinstance(flushed, list):
            flushed = [flushed] if flushed is not None else []
        for out in flushed:
            consume(out.to_ndarray())
    return [float(value) for value in np.clip(peaks, 0.0, 1.0)]


def _target_size(width: Any = None, height: Any = None) -> tuple[int, int] | None:
    """Validate a generation canvas size used to bound decoded reference media."""

    target_width, target_height = int(_float(width)), int(_float(height))
    if target_width < 1 or target_height < 1:
        return None
    return target_width, target_height


def _fit_image(image: Image.Image, size: tuple[int, int], *, video: bool = False) -> Image.Image:
    """Aspect-fill a reference into the generation canvas without distortion."""

    method = Image.Resampling.BILINEAR if video else Image.Resampling.LANCZOS
    return ImageOps.fit(image.convert("RGB"), size, method=method, centering=(0.5, 0.5))


def _decode_video(
    path: Path,
    start: float,
    duration: float,
    fps: float = FPS,
    target_width: Any = None,
    target_height: Any = None,
) -> torch.Tensor | None:
    start = max(0.0, float(start))
    duration = max(0.0, float(duration))
    if duration <= 0:
        return None
    target_count = max(1, int(round(duration * fps)))
    target_times = [start + i / fps for i in range(target_count)]
    decoded: list[tuple[float, np.ndarray]] = []
    target_size = _target_size(target_width, target_height)
    with av.open(str(path)) as container:
        if not container.streams.video:
            return None
        stream = container.streams.video[0]
        seek_to = max(0.0, start - 1.0)
        if stream.time_base:
            container.seek(int(seek_to / float(stream.time_base)), stream=stream, backward=True)
        else:
            container.seek(int(seek_to * av.time_base), backward=True)
        limit = start + duration + 1.0 / fps
        fallback_time = seek_to
        source_fps = float(stream.average_rate or stream.guessed_rate or fps)
        for index, frame in enumerate(container.decode(stream)):
            timestamp = float(frame.time) if frame.time is not None else fallback_time + index / source_fps
            if timestamp < start - 1.0 / fps:
                continue
            if timestamp > limit:
                break
            if target_size:
                # Resize each decoded frame immediately.  This is deliberately
                # done before frames are retained, so a 4K/8K source never
                # accumulates a full-resolution reference tensor in RAM/VRAM.
                array = np.asarray(_fit_image(frame.to_image(), target_size, video=True), dtype=np.uint8)
            else:
                array = frame.to_ndarray(format="rgb24")
            decoded.append((timestamp, np.ascontiguousarray(array)))
    if not decoded:
        return None
    frames: list[np.ndarray] = []
    cursor = 0
    for target in target_times:
        while cursor + 1 < len(decoded) and abs(decoded[cursor + 1][0] - target) <= abs(decoded[cursor][0] - target):
            cursor += 1
        frames.append(decoded[cursor][1])
    array = np.stack(frames).astype(np.float32) / 255.0
    return torch.from_numpy(array)


def _decode_frame(
    path: Path, second: float, target_width: Any = None, target_height: Any = None
) -> torch.Tensor | None:
    frames = _decode_video(
        path, max(0.0, second), 1.0 / FPS, FPS, target_width, target_height
    )
    return None if frames is None else frames[:1]


def _load_image(path: Path, target_width: Any = None, target_height: Any = None) -> torch.Tensor:
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
        target_size = _target_size(target_width, target_height)
        if target_size:
            image = _fit_image(image, target_size)
        arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)


def _aligned_h3_length(seconds: float) -> int:
    """Return the nearest H3-valid 5 + 17*n frame count."""

    requested = max(5.0 / FPS, float(seconds)) * FPS
    n = max(0, round((requested - 5.0) / 17.0))
    return int(5 + 17 * n)


def _windows(start: float, duration: float, limit: float = MAX_REF_VIDEO_SECONDS) -> Iterable[tuple[float, float]]:
    remaining = max(0.0, duration)
    cursor = start
    while remaining >= MIN_REF_VIDEO_SECONDS:
        piece = min(limit, remaining)
        yield cursor, piece
        cursor += piece
        remaining -= piece


def _clip_end(clip: dict[str, Any]) -> float:
    return _float(clip.get("start")) + max(0.0, _float(clip.get("duration")))


def _source_time(clip: dict[str, Any], timeline_second: float) -> float:
    return max(0.0, _float(clip.get("trimStart")) + timeline_second - _float(clip.get("start")))


def _reference_mode(clip: dict[str, Any]) -> str:
    mode = str(clip.get("referenceMode") or "guide")
    return mode if mode in {"guide", "edit", "boundary"} else "guide"


def _video_reference_specs(timeline: dict[str, Any]) -> list[dict[str, Any]]:
    """Describe prompt-addressable videos without decoding their frames.

    This is shared by the material planner and the H3 encoder, so ``Video N``
    exposed to a prompt rewriter is guaranteed to be the same source interval
    later encoded as ``ref_video_N``.  A clip contributes only its intersection
    with the cyan generation selection; frames outside that intersection never
    enter prompt rewriting, reference encoding, or per-clip guides.
    """

    clips = [c for c in timeline.get("videoClips", []) if isinstance(c, dict) and c.get("file")]
    selection = timeline.get("selection") or {}
    selection_start = max(0.0, _float(selection.get("start")))
    selection_duration = max(MIN_REF_VIDEO_SECONDS, _float(selection.get("duration"), 5.0))
    selection_end = selection_start + selection_duration
    audio_enabled = timeline.get("videoAudioEnabled", True) is not False
    specs: list[dict[str, Any]] = []

    for clip in sorted(clips, key=lambda c: _float(c.get("start"))):
        clip_start, clip_end = _float(clip.get("start")), _clip_end(clip)
        overlap_start = max(selection_start, clip_start)
        overlap_end = min(selection_end, clip_end)
        if overlap_end - overlap_start <= 0:
            continue

        overlap_source_start = _source_time(clip, overlap_start)
        for source_start, piece_duration in _windows(
            overlap_source_start, overlap_end - overlap_start
        ):
            timeline_piece_start = overlap_start + source_start - overlap_source_start
            specs.append({
                "file": str(clip["file"]),
                "name": str(clip.get("name") or Path(str(clip["file"])).name),
                "clip_id": clip.get("id"),
                "timeline_start": timeline_piece_start,
                "timeline_end": timeline_piece_start + piece_duration,
                "source_start": source_start,
                "duration": piece_duration,
                "has_audio": audio_enabled and clip.get("hasAudio", True) is not False,
            })
            if len(specs) >= MAX_REF_VIDEOS:
                return specs
    return specs


def _ordered_segment_assets(
    assets: list[Any], requested_ids: Any, limit: int
) -> list[dict[str, Any]]:
    """Return one segment's assets in its explicit drag order.

    Segment membership is stored by stable UI ids instead of list positions, so
    deleting or reordering the global upload library cannot silently point a
    segment at a different file.
    """

    by_id = {
        str(asset.get("id")): asset
        for asset in assets
        if isinstance(asset, dict) and asset.get("id") and asset.get("file")
    }
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in requested_ids if isinstance(requested_ids, list) else []:
        asset_id = str(value)
        asset = by_id.get(asset_id)
        if asset is not None and asset_id not in seen:
            ordered.append(asset)
            seen.add(asset_id)
            if len(ordered) >= limit:
                break
    return ordered


def _timeline_for_prompt_index(
    timeline: dict[str, Any], prompt_index: Any = None
) -> tuple[dict[str, Any], int | None, int]:
    """Apply the optional 1-based prompt/segment material assignment.

    With no connected prompt index, or with segment count zero, the legacy
    unsegmented timeline is returned unchanged.  Once segmentation is enabled,
    independent images and audio are filtered and ordered per segment; all
    downstream Picture/Audio labels therefore restart from one automatically.
    """

    result = copy.deepcopy(timeline)
    config = result.get("segmentConfig")
    if not isinstance(config, dict):
        config = {}
    raw_segment_count = int(_float(config.get("count"), 0))
    if raw_segment_count < 1:
        # Version 6 makes one GEN window the minimum.  Preserve old workflows
        # which stored count=0 by turning the visible selection into segment 1
        # and assigning the former unsegmented references to it.
        selection = result.get("selection") or {}
        duration = max(MIN_REF_VIDEO_SECONDS, _float(selection.get("duration"), 5.0))
        image_ids: list[str] = []
        for index, asset in enumerate(result.get("images", [])):
            if not isinstance(asset, dict) or not asset.get("file"):
                continue
            asset.setdefault("id", f"legacy-image-{index + 1}")
            image_ids.append(str(asset["id"]))
        audio_ids: list[str] = []
        for index, asset in enumerate(result.get("audios", [])):
            if not isinstance(asset, dict) or not asset.get("file"):
                continue
            asset.setdefault("id", f"legacy-audio-{index + 1}")
            audio_ids.append(str(asset["id"]))
        image_ids = image_ids[:MAX_REF_IMAGES]
        audio_ids = audio_ids[:MAX_REF_AUDIOS]
        config = {
            **config,
            "count": 1,
            "activeIndex": 0,
            "mode": "timeline",
            "segments": [{
                "startFrame": 0,
                "endFrame": _aligned_h3_length(duration),
                "images": image_ids,
                "audios": audio_ids,
                "prompt": "",
            }],
        }
        result["segmentConfig"] = config
    segment_count = max(1, int(_float(config.get("count"), 1)))
    if prompt_index is None:
        return result, None, segment_count

    selected_index = max(1, int(_float(prompt_index, 1)))
    segments = config.get("segments")
    if not isinstance(segments, list) or selected_index > segment_count:
        raise ValueError(
            f"Prompt index {selected_index} exceeds the planner's {segment_count} material segments; "
            "keep the iteration count, prompt segment count, and planner segment count identical."
        )
    while len(segments) < segment_count:
        segments.append({})
    selected = segments[selected_index - 1]
    if not isinstance(selected, dict):
        selected = {}

    result["images"] = _ordered_segment_assets(
        list(result.get("images") or []), selected.get("images"), MAX_REF_IMAGES
    )
    result["audios"] = _ordered_segment_assets(
        list(result.get("audios") or []), selected.get("audios"), MAX_REF_AUDIOS
    )
    if config.get("mode") == "timeline":
        start, end = int(selected.get("startFrame", 0)), int(selected.get("endFrame", 5))
        result["selection"] = {"start": start / FPS, "duration": (end - start) / FPS}
    return result, selected_index, segment_count


def _create_timeline_plan(
    timeline_data: str, width: Any, height: Any, generation_seconds: Any,
    prompt_index: Any = None,
) -> dict[str, Any]:
    """Create the lightweight, serializable half of the split-node workflow."""

    timeline, selected_segment, segment_count = _timeline_for_prompt_index(
        _parse_timeline(timeline_data), prompt_index
    )
    target_width, target_height = int(width), int(height)
    seconds = max(MIN_REF_VIDEO_SECONDS, _float(generation_seconds, 5.0))
    if selected_segment is not None and timeline.get("segmentConfig", {}).get("mode") == "timeline":
        seconds = timeline["selection"]["duration"]
    selection = timeline.setdefault("selection", {})
    selection["start"] = max(0.0, _float(selection.get("start")))
    # The visible duration widget is authoritative.  Persisting it into the
    # plan also prevents a stale hidden timeline value from shifting guides.
    selection["duration"] = seconds
    return {
        "type": "MINIMAX_H3_TIMELINE_PLAN",
        "version": 1,
        "timeline": timeline,
        "width": target_width,
        "height": target_height,
        "generation_seconds": seconds,
        "length": _aligned_h3_length(seconds),
        "prompt_index": selected_segment,
        "segment_count": segment_count,
    }


def _require_timeline_plan(plan: Any) -> dict[str, Any]:
    if not isinstance(plan, dict) or plan.get("type") != "MINIMAX_H3_TIMELINE_PLAN":
        raise ValueError("输入不是合法的 MiniMax H3 时间线素材计划。")
    if not isinstance(plan.get("timeline"), dict):
        raise ValueError("素材计划里没有时间线数据。")
    return plan


def _preview_canvas(width: int, height: int, edge: int = 480) -> tuple[int, int]:
    width, height = max(1, int(width)), max(1, int(height))
    scale = min(1.0, edge / width, edge / height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _plan_prompt_media(plan: dict[str, Any]) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    """Materialize only the small media surfaces needed by prompt rewriters."""

    plan = _require_timeline_plan(plan)
    timeline = plan["timeline"]
    preview_width, preview_height = _preview_canvas(plan["width"], plan["height"])

    pictures: list[Any] = []
    for asset in timeline.get("images", []):
        if len(pictures) >= MAX_REF_IMAGES:
            break
        if isinstance(asset, dict) and asset.get("file"):
            pictures.append(_load_image(
                _safe_input_path(str(asset["file"])), preview_width, preview_height
            ))

    video_specs = _video_reference_specs(timeline)
    videos: list[Any] = [
        TrimmedTimelineVideo(
            str(_safe_input_path(spec["file"])),
            float(spec["source_start"]),
            float(spec["duration"]),
        )
        for spec in video_specs
    ]

    standalone_audios: list[Any] = []
    for asset in timeline.get("audios", []):
        if len(standalone_audios) >= MAX_REF_AUDIOS:
            break
        if not isinstance(asset, dict) or not asset.get("file"):
            continue
        if _audio_mode(asset) == "locked":
            continue
        audio = _decode_audio(
            _safe_input_path(str(asset["file"])),
            max(0.0, _float(asset.get("trimStart"))),
            _float(asset.get("duration")) or None,
        )
        if audio is not None:
            standalone_audios.append(audio)

    paired_audios: list[Any] = []
    for spec in video_specs:
        if not spec["has_audio"]:
            paired_audios.append(None)
            continue
        paired_audios.append(_decode_audio(
            _safe_input_path(spec["file"]), spec["source_start"], spec["duration"]
        ))

    return pictures, videos, standalone_audios, paired_audios


def _create_prompt_media_bundle(plan: dict[str, Any]) -> dict[str, Any]:
    """Package ordered heterogeneous media behind one compact ComfyUI socket."""

    plan = _require_timeline_plan(plan)
    pictures, videos, standalone_audios, paired_audios = _plan_prompt_media(plan)
    items: list[dict[str, Any]] = []
    for index, value in enumerate(pictures, 1):
        items.append({"kind": "image", "label": f"<Picture {index}>", "value": value})
    for index, value in enumerate(videos, 1):
        items.append({"kind": "video", "label": f"<Video {index}>", "value": value})

    audio_index = 0
    for value in standalone_audios:
        audio_index += 1
        items.append({"kind": "audio", "label": f"<Audio {audio_index}>", "value": value})
    for value in paired_audios:
        if value is None:
            continue
        audio_index += 1
        items.append({"kind": "audio", "label": f"<Audio {audio_index}>", "value": value})

    return {
        "type": "MINIMAX_H3_OMNI_MEDIA_BUNDLE",
        "version": 1,
        "items": items,
        "manifest": _reference_manifest(plan),
        "generation_seconds": float(plan["generation_seconds"]),
        "width": int(plan["width"]),
        "height": int(plan["height"]),
    }


def _require_prompt_media_bundle(bundle: Any) -> dict[str, Any]:
    if not isinstance(bundle, dict) or bundle.get("type") != "MINIMAX_H3_OMNI_MEDIA_BUNDLE":
        raise ValueError("输入不是合法的 MiniMax H3 全媒体素材包。")
    items = bundle.get("items")
    if not isinstance(items, list):
        raise ValueError("全媒体素材包里没有排好序的媒体列表。")
    return bundle


def _reference_manifest(plan: dict[str, Any]) -> str:
    """供提示词编写节点与 AI 代理共用的可读性标签表。"""

    plan = _require_timeline_plan(plan)
    timeline = plan["timeline"]
    lines = [
        "MiniMax H3 时间线素材计划（标签顺序与 H3 编码器完全一致）",
        f"Target: {plan['width']}x{plan['height']}, {plan['generation_seconds']:.3f}s, {plan['length']} frames",
    ]
    picture_index = 0
    for asset in timeline.get("images", []):
        if picture_index >= MAX_REF_IMAGES:
            break
        if isinstance(asset, dict) and asset.get("file"):
            picture_index += 1
            lines.append(f"<Picture {picture_index}> = {asset.get('name') or Path(str(asset['file'])).name}")

    video_specs = _video_reference_specs(timeline)
    for index, spec in enumerate(video_specs, 1):
        end = spec["source_start"] + spec["duration"]
        lines.append(
            f"<Video {index}> = {spec['name']}，源片段 {spec['source_start']:.3f}s–{end:.3f}s"
        )

    audio_index = 0
    for asset in timeline.get("audios", []):
        if audio_index >= MAX_REF_AUDIOS:
            break
        if (
            isinstance(asset, dict) and asset.get("file")
            and _audio_mode(asset) != "locked"
        ):
            audio_index += 1
            lines.append(f"<Audio {audio_index}> = 独立音频 {asset.get('name') or Path(str(asset['file'])).name}")
    locked = _locked_audio_assets(timeline)
    if locked:
        lines.append(
            "锁定音轨 = "
            + ", ".join(str(asset.get("name") or Path(str(asset["file"])).name) for asset in locked)
            + "（不占用 <Audio N> 参考标签）"
        )
    for video_index, spec in enumerate(video_specs, 1):
        if spec["has_audio"]:
            audio_index += 1
            lines.append(f"<Audio {audio_index}> = 与 <Video {video_index}> 配对的原始音频")

    total_rewriter_media = picture_index + len(video_specs) + audio_index
    if total_rewriter_media > 12:
        lines.append(
            f"提示：当前共有 {total_rewriter_media} 个媒体参考；Prompt Rewriter Omni 最多接受 12 个。请只保留理解该提示词所必需的媒体。"
        )

    if not video_specs:
        lines.append("当前没有可供提示词寻址的 Video 参考；固定 Guides 与自动边界帧不占用 Picture/Video 序号。")
    else:
        lines.append("Video 序号严格从左到右排列，且每个 Video 只包含它与生成区间的重叠部分。")
        lines.append("固定 Guides 复用这部分重叠；自动空隙边界帧不占用 Picture/Video 序号。")
    return "\n".join(lines)


def _pick_gap_guides(clips: list[dict[str, Any]], selection_start: float, selection_end: float) -> list[tuple[dict[str, Any], float, int, str]]:
    """Choose still guides only when the selection is a true empty gap."""

    if not clips:
        return []
    epsilon = 1.0 / FPS
    ordered = sorted(clips, key=lambda clip: (_float(clip.get("start")), _clip_end(clip)))
    if any(min(selection_end, _clip_end(c)) - max(selection_start, _float(c.get("start"))) > epsilon for c in ordered):
        return []
    result: list[tuple[dict[str, Any], float, int, str]] = []
    left = [c for c in ordered if _clip_end(c) <= selection_start + epsilon]
    if left:
        clip = max(left, key=_clip_end)
        result.append((clip, max(_float(clip.get("start")), _clip_end(clip) - epsilon), 0, "gap-start"))
    right = [c for c in ordered if _float(c.get("start")) >= selection_end - epsilon]
    if right:
        clip = min(right, key=lambda c: _float(c.get("start")))
        result.append((clip, _float(clip.get("start")), -1, "gap-end"))
    return result


def _valid_guide_frame_count(frame_count: int) -> int:
    """Mirror MiniMaxH3AddGuide's legal 1 or 17*k+5 frame rule."""

    frame_count = max(1, int(frame_count))
    if frame_count < 5:
        return 1
    return frame_count - ((frame_count - 5) % 17)


def _guide_frame_slices(frame_count: int) -> list[tuple[int, int]]:
    """Split every source frame into legal native guides without dropping tails."""

    result: list[tuple[int, int]] = []
    offset = 0
    remaining = max(0, int(frame_count))
    while remaining:
        count = _valid_guide_frame_count(remaining)
        result.append((offset, count))
        offset += count
        remaining -= count
    return result


def _parse_timeline(value: str) -> dict[str, Any]:
    try:
        data = json.loads(value or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"时间线数据不是合法的 JSON：{exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("时间线数据必须是 JSON 对象")
    return data


def _build_references(
    timeline: dict[str, Any], target_width: Any = None, target_height: Any = None,
    target_length: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    clips = [c for c in timeline.get("videoClips", []) if isinstance(c, dict) and c.get("file")]
    selection = timeline.get("selection") or {}
    selection_start = max(0.0, _float(selection.get("start")))
    selection_duration = max(MIN_REF_VIDEO_SECONDS, _float(selection.get("duration"), 5.0))
    selection_end = selection_start + selection_duration
    target_length = int(target_length or _aligned_h3_length(selection_duration))

    ref_images: dict[str, torch.Tensor] = {}
    ref_videos: dict[str, torch.Tensor] = {}
    ref_video_audios: dict[str, dict[str, Any]] = {}
    ref_audios: dict[str, dict[str, Any]] = {}
    guides: list[dict[str, Any]] = []
    video_audio_enabled = timeline.get("videoAudioEnabled", True) is not False

    # User-uploaded bins own their visible ordinal space. Native guides are not
    # prompt-labelled Picture references and therefore never change ordinals.
    for asset in timeline.get("images", []):
        if len(ref_images) >= MAX_REF_IMAGES:
            break
        if not isinstance(asset, dict) or not asset.get("file"):
            continue
        ref_images[f"ref_image_{len(ref_images)}"] = _load_image(
            _safe_input_path(str(asset["file"])), target_width, target_height
        )

    # Decode the exact intervals advertised by the planning node.  Keeping this
    # as one shared plan guarantees Video/Audio ordinals cannot drift between
    # prompt rewriting and the final H3 encode.
    for spec in _video_reference_specs(timeline):
        reference_frames = _decode_video(
            _safe_input_path(spec["file"]), spec["source_start"], spec["duration"],
            FPS, target_width, target_height,
        )
        if reference_frames is None or reference_frames.shape[0] < 5:
            continue
        index = len(ref_videos)
        ref_videos[f"ref_video_{index}"] = reference_frames
        if video_audio_enabled and spec["has_audio"]:
            audio = _decode_audio(
                _safe_input_path(spec["file"]), spec["source_start"], spec["duration"]
            )
            if audio is not None:
                ref_video_audios[f"ref_video_audio_{index}"] = audio

    # A clip crossing the generated range is split deliberately: the overlap is
    # a hard guide at its target position; only the source context outside the
    # range is a prompt-addressable Video reference.
    for clip in sorted(clips, key=lambda c: _float(c.get("start"))):
        clip_start, clip_end = _float(clip.get("start")), _clip_end(clip)
        overlap_start = max(selection_start, _float(clip.get("start")))
        overlap_end = min(selection_end, clip_end)
        overlap_duration = overlap_end - overlap_start
        if overlap_duration <= 0:
            continue
        path = _safe_input_path(str(clip["file"]))
        mode = _reference_mode(clip)

        if mode in {"edit", "boundary"}:
            if mode == "boundary":
                first_idx = max(0, round((overlap_start - selection_start) * FPS))
                last_idx = min(target_length - 1, max(first_idx, round((overlap_end - selection_start) * FPS) - 1))
                if abs(overlap_start - selection_start) <= 0.5 / FPS:
                    first_idx = 0
                if abs(overlap_end - selection_end) <= 0.5 / FPS:
                    last_idx = target_length - 1
                first_frame = _decode_frame(
                    path, _source_time(clip, overlap_start), target_width, target_height
                )
                if first_frame is not None:
                    guides.append({
                        "image": first_frame, "audio": None, "frame_idx": first_idx,
                        "kind": "boundary-start", "clip_id": clip.get("id"), "source_frames": 1,
                    })
                if last_idx != first_idx:
                    last_frame = _decode_frame(
                        path, _source_time(clip, max(overlap_start, overlap_end - 1.0 / FPS)),
                        target_width, target_height,
                    )
                    if last_frame is not None:
                        guides.append({
                            "image": last_frame, "audio": None, "frame_idx": last_idx,
                            "kind": "boundary-end", "clip_id": clip.get("id"), "source_frames": 1,
                        })
            continue

        total_frames = min(max(1, round(overlap_duration * FPS)), target_length)
        guide_start_frame = max(0, round((overlap_start - selection_start) * FPS))
        if abs(overlap_start - selection_start) <= 0.5 / FPS:
            guide_start_frame = 0
        elif abs(overlap_end - selection_end) <= 0.5 / FPS:
            guide_start_frame = max(0, target_length - total_frames)
        guide_start_frame = min(guide_start_frame, max(0, target_length - total_frames))

        # Decode long overlaps in bounded windows. This preserves the existing
        # resolution protection without ever materializing a 150-second tensor.
        decoded_frames = 0
        while decoded_frames < total_frames:
            requested_frames = min(
                round(MAX_REF_VIDEO_SECONDS * FPS), total_frames - decoded_frames
            )
            frames = _decode_video(
                path, _source_time(clip, overlap_start) + decoded_frames / FPS,
                requested_frames / FPS, FPS, target_width, target_height,
            )
            if frames is None or not frames.shape[0]:
                break
            chunk_frames = min(int(frames.shape[0]), requested_frames)
            frames = frames[:chunk_frames]
            for slice_offset, slice_count in _guide_frame_slices(chunk_frames):
                guide_audio = None
                if video_audio_enabled and clip.get("hasAudio", True):
                    guide_audio = _decode_audio(
                        path, _source_time(clip, overlap_start) + (decoded_frames + slice_offset) / FPS,
                        slice_count / FPS,
                    )
                guides.append({
                    "image": frames[slice_offset:slice_offset + slice_count],
                    "audio": guide_audio,
                    "frame_idx": guide_start_frame + decoded_frames + slice_offset,
                    "kind": "video", "clip_id": clip.get("id"),
                    "source_frames": slice_count,
                })
            decoded_frames += chunk_frames
            if chunk_frames < requested_frames:
                break

    # A true gap has no video reference. Its nearest visual context is anchored
    # as target first/last frames with the official Add Guide mechanism.
    for clip, timeline_second, frame_idx, kind in _pick_gap_guides(clips, selection_start, selection_end):
        frame = _decode_frame(
            _safe_input_path(str(clip["file"])), _source_time(clip, timeline_second),
            target_width, target_height,
        )
        if frame is not None:
            guides.append({"image": frame, "audio": None, "frame_idx": 0 if frame_idx == 0 else target_length - 1, "kind": kind})

    for asset in timeline.get("audios", []):
        if len(ref_audios) >= MAX_REF_AUDIOS:
            break
        if not isinstance(asset, dict) or not asset.get("file"):
            continue
        # Locked soundtrack assets are target AV content, not prompt-addressable
        # H3 reference audio. Finite Segment Sampling injects their encoded slice
        # into the target audio stream and fixes its denoise mask to zero.
        if _audio_mode(asset) == "locked":
            continue
        start = max(0.0, _float(asset.get("trimStart")))
        duration = _float(asset.get("duration")) or None
        audio = _decode_audio(_safe_input_path(str(asset["file"])), start, duration)
        if audio is not None:
            ref_audios[f"ref_audio_{len(ref_audios)}"] = audio

    return ref_images, ref_videos, ref_video_audios, ref_audios, guides


def _apply_h3_guides(conditioning, latent, vae, audio_vae, guides: list[dict[str, Any]]):
    """Apply timeline guides through ComfyUI's native, versioned H3 node."""

    if not guides:
        return conditioning
    if not hasattr(h3_nodes, "MiniMaxH3AddGuide"):
        raise RuntimeError("当前 ComfyUI 版本没有 MiniMaxH3AddGuide 节点，请升级到包含 PR #15439 的版本。")
    for guide in guides:
        conditioning = h3_nodes.MiniMaxH3AddGuide.execute(
            conditioning, latent, int(guide["frame_idx"]), vae=vae,
            audio_vae=audio_vae, image=guide.get("image"), audio=guide.get("audio"),
        )[0]
    return conditioning


def _execute_h3_independent_first(
    clip, vae, audio_vae, prompt: str, width: int, height: int, length: int,
    ref_image_size: str, ref_images: dict[str, torch.Tensor],
    ref_videos: dict[str, torch.Tensor], ref_video_audios: dict[str, dict[str, Any]],
    ref_audios: dict[str, dict[str, Any]],
) -> tuple[Any, Any]:
    """H3 Ref2VA with standalone Audio ordinals before paired video soundtracks.

    The official node presents video soundtracks before standalone audio.  The
    director intentionally presents independent audio first so the visible bin
    is stable as <Audio 1>, <Audio 2>, ... regardless of timeline video refs.
    """

    latent, frame_count = h3_nodes._empty_av_latent(width, height, length)
    ref_items: list[dict[str, Any]] = []
    ref_blocks: list[dict[str, Any]] = []

    for image in (ref_images or {}).values():
        if image is None:
            continue
        image_height, image_width = image.shape[1], image.shape[2]
        if ref_image_size == "match":
            scale = min(1.0, math.sqrt((width * height) / (image_width * image_height)))
        else:
            scale = min(1.0, h3_nodes.REF_IMAGE_SHORT_EDGE / min(image_width, image_height))
        target_width = max(
            h3_nodes.CANVAS_MULTIPLE,
            round(image_width * scale / h3_nodes.CANVAS_MULTIPLE) * h3_nodes.CANVAS_MULTIPLE,
        )
        target_height = max(
            h3_nodes.CANVAS_MULTIPLE,
            round(image_height * scale / h3_nodes.CANVAS_MULTIPLE) * h3_nodes.CANVAS_MULTIPLE,
        )
        resized = h3_nodes._resize(image[:1], target_width, target_height, "disabled")
        encoded = vae.encode(resized)
        ref_items.append({"type": "image", "data": resized})
        ref_blocks.append({
            "kind": "image", "latent_h": target_height // 16,
            "latent_w": target_width // 16, "latent": encoded,
        })

    # Standalone references own Audio 1..N.
    for audio in (ref_audios or {}).values():
        if audio is None:
            continue
        audio_latent, ref_audio_t = h3_nodes._encode_ref_audio(audio_vae, audio)
        ref_items.append({"type": "audio"})
        ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})

    ref_video_audios = ref_video_audios or {}
    for name, video_frames in (ref_videos or {}).items():
        if video_frames is None:
            continue
        soundtrack = ref_video_audios.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
        video_height, video_width = video_frames.shape[1], video_frames.shape[2]
        canvas_width, canvas_height = h3_nodes.adapt_canvas(video_width, video_height)
        if video_width * video_height < canvas_width * canvas_height:
            canvas_width = max(
                h3_nodes.CANVAS_MULTIPLE,
                round(video_width / h3_nodes.CANVAS_MULTIPLE) * h3_nodes.CANVAS_MULTIPLE,
            )
            canvas_height = max(
                h3_nodes.CANVAS_MULTIPLE,
                round(video_height / h3_nodes.CANVAS_MULTIPLE) * h3_nodes.CANVAS_MULTIPLE,
            )
        frames = h3_nodes._resize(video_frames, canvas_width, canvas_height, "disabled")
        if frames.shape[0] > frame_count:
            frames = frames[:frame_count]
        frame_total = frames.shape[0]
        if frame_total < 5:
            raise ValueError("MiniMax H3 参考视频至少需要 5 帧（24 fps 下约 0.2 秒）")
        while frame_total % 17 != 5:
            frame_total -= 1
        frames = frames[:frame_total]
        video_latent = vae.encode(frames)
        audio_latent, ref_audio_t = (None, 0)
        if soundtrack is not None:
            audio_latent, ref_audio_t = h3_nodes._encode_ref_audio(audio_vae, soundtrack)
            ref_items.append({"type": "audio"})
        sample_indices = list(range(0, frames.shape[0], h3_nodes.FPS // 2))
        qwen_frames = frames[sample_indices]
        ref_items.append({
            "type": "video", "data": qwen_frames,
            "timestamps": [index / 2.0 for index in range(len(sample_indices))],
        })
        ref_blocks.append({
            "kind": "video_audio" if ref_audio_t else "video",
            "latent_t": video_latent.shape[2], "latent_h": canvas_height // 16,
            "latent_w": canvas_width // 16, "ref_audio_t": ref_audio_t,
            "latent": video_latent, "audio_latent": audio_latent,
        })

    tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    if ref_blocks:
        conditioning = node_helpers.conditioning_set_values(
            conditioning, {"minimax_refs": ref_blocks}
        )
    return conditioning, latent


async def _acquire_http_slot(semaphore: asyncio.Semaphore) -> bool:
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=2.0)
        return True
    except asyncio.TimeoutError:
        return False


async def _request_json(request: web.Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise ValueError("请求体不是合法的 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("请求体必须是 JSON 对象")
    return payload


@PromptServer.instance.routes.post("/minimax_h3_timeline/media_info")
async def media_info(request: web.Request) -> web.Response:
    acquired = False
    try:
        payload = await _request_json(request)
        kind = str(payload.get("kind") or "").lower()
        if kind not in MEDIA_EXTENSIONS:
            raise ValueError("媒体类型只能是图片、音频或视频")
        path = _safe_uploaded_path(str(payload.get("filename") or ""), kind)
        acquired = await _acquire_http_slot(_MEDIA_INFO_SEMAPHORE)
        if not acquired:
            return web.json_response({"error": "媒体检查正忙，请稍后重试"}, status=429)
        info = await asyncio.get_running_loop().run_in_executor(
            None, _uploaded_media_info, path, kind
        )
        return web.json_response(info)
    except FileNotFoundError:
        return web.json_response({"error": "找不到上传的媒体"}, status=404)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except Exception:
        log.exception("Media inspection failed")
        return web.json_response({"error": "无法检查上传的媒体"}, status=400)
    finally:
        if acquired:
            _MEDIA_INFO_SEMAPHORE.release()


@PromptServer.instance.routes.post("/minimax_h3_timeline/preview_proxy")
async def preview_proxy(request: web.Request) -> web.Response:
    """Return (and lazily create) the cached low-resolution monitoring proxy."""

    acquired = False
    try:
        payload = await _request_json(request)
        path = _safe_uploaded_path(str(payload.get("filename") or ""), "video")
        acquired = await _acquire_http_slot(_PREVIEW_SEMAPHORE)
        if not acquired:
            return web.json_response({"error": "预览生成正忙，请稍后重试"}, status=429)
        proxy = await asyncio.get_running_loop().run_in_executor(None, _ensure_preview_proxy, path)
        return web.json_response({
            "proxy": proxy,
            "width": PREVIEW_MAX_WIDTH,
            "height": PREVIEW_MAX_HEIGHT,
            "fps": PREVIEW_FPS,
        })
    except FileNotFoundError:
        return web.json_response({"error": "找不到上传的视频"}, status=404)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except Exception:
        log.exception("Preview proxy failed")
        return web.json_response({"error": "无法生成预览视频"}, status=400)
    finally:
        if acquired:
            _PREVIEW_SEMAPHORE.release()


def _encode_timeline_plan(
    plan: dict[str, Any], clip, vae, audio_vae, prompt: str, ref_image_size: str
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    plan = _require_timeline_plan(plan)
    timeline = plan["timeline"]
    target_width, target_height = int(plan["width"]), int(plan["height"])
    length = int(plan["length"])
    ref_images, ref_videos, ref_video_audios, ref_audios, guides = _build_references(
        timeline, target_width, target_height, length
    )
    if not (ref_images or ref_videos or ref_audios or guides):
        log.info(
            "No reference materials in the timeline plan; encoding MiniMax H3 as pure text-to-video."
        )
    video_audio_output = _timeline_video_audio(timeline)
    standalone_audio_output = _standalone_audio_track(timeline)
    log.info(
        "Encoding planned H3 refs at %dx%d: %d images, %d videos, %d paired audios, %d standalone audios, %d native guides; %d output frames",
        target_width, target_height, len(ref_images), len(ref_videos),
        len(ref_video_audios), len(ref_audios), len(guides), length,
    )
    conditioning, latent = _execute_h3_independent_first(
        clip, vae, audio_vae, prompt, target_width, target_height, length,
        ref_image_size, ref_images, ref_videos, ref_video_audios, ref_audios,
    )
    conditioning = _apply_h3_guides(conditioning, latent, vae, audio_vae, guides)
    return conditioning, latent, video_audio_output, standalone_audio_output


class MiniMaxH3TimelinePlanner(io.ComfyNode):
    """Editable material/guide plan which deliberately performs no H3 encode."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WJZ_H3_TimelinePlanner",
            display_name="H3 素材与分段计划台",
            description=(
                "编辑一条时间线并输出一份轻量素材计划。这份计划可以先喂给提示词改写器，"
                "改写后的提示词与同一份计划再进入 H3 分段编码器，避免形成回路。"
            ),
            category="model/conditioning/minimax",
            inputs=[
                io.Int.Input(
                    "prompt_index", display_name="分段序号", optional=True,
                    force_input=True, tooltip=(
                        "可选。接一个分段序号，则只输出分配给该分段的图片与独立音频；"
                        "不接则使用全部素材。"
                    ),
                ),
                io.Int.Input(
                    "width", display_name="宽度", default=1344, min=32, max=16384, step=32,
                    tooltip="输出画面的像素宽度，与时间线共用同一组约束。",
                ),
                io.Int.Input(
                    "height", display_name="高度", default=768, min=32, max=16384, step=32,
                    tooltip="输出画面的像素高度，与时间线共用同一组约束。",
                ),
                io.Float.Input(
                    "generation_seconds", display_name="生成时长（秒）",
                    default=5.0, min=0.21, max=150.0, step=0.1,
                    tooltip="生成时长，与青色时间线区间双向同步。",
                ),
                io.String.Input(
                    "timeline_data", display_name="时间线数据", default="", multiline=True,
                    tooltip="由「H3 导演台」导出的时间线文本，可在此继续编辑。",
                ),
            ],
            outputs=[
                TimelinePlan.Output(display_name="素材计划"),
                PromptMediaBundle.Output(display_name="全媒体素材包"),
                io.Custom("MINIMAX_H3_FINITE_SEGMENT_PLAN").Output(display_name="分段计划"),
            ],
        )

    @classmethod
    def execute(
        cls, width, height, generation_seconds, timeline_data="", prompt_index=None
    ) -> io.NodeOutput:
        complete_plan = _create_timeline_plan(
            timeline_data, width, height, generation_seconds, None
        )
        # The sampler validates prompts only when this output is actually used.
        # Existing single-segment encoder/Omni workflows remain usable while editing.
        selected_plan = complete_plan
        config = complete_plan["timeline"].get("segmentConfig", {})
        if prompt_index is not None:
            selected_plan = _create_timeline_plan(
                timeline_data, width, height, generation_seconds, prompt_index,
            )
        elif config.get("mode") == "timeline" and int(config.get("count", 0)) > 0:
            selected_plan = _create_timeline_plan(
                timeline_data, width, height, generation_seconds,
                int(config.get("activeIndex", 0)) + 1,
            )
        return io.NodeOutput(
            selected_plan, _create_prompt_media_bundle(selected_plan), complete_plan
        )


class MiniMaxH3OmniPromptBridge(io.ComfyNode):
    """Run Prompt Rewriter Omni directly from the planner's ordered media bundle."""

    @classmethod
    def define_schema(cls):
        tasks, models, quantizations = _omni_schema_choices()
        default_task = "REF2AV" if "REF2AV" in tasks else tasks[0]
        return io.Schema(
            node_id="WJZ_H3_OmniPromptBridge",
            display_name="H3 全媒体提示词桥接",
            description=(
                "直接读取计划台排好序的全媒体素材包，调用 MiniMax-H3 提示词改写器 Omni 后端，"
                "无需逐个展开、连线各类媒体端口。"
            ),
            category="MiniMax-H3",
            inputs=[
                PROMPT_REWRITER_OPTIONS.Input(
                    "options", display_name="改写器选项", optional=True,
                    tooltip="透传给 Prompt Rewriter Omni 后端的额外选项字典，留空则使用该后端默认值。",
                ),
                PromptMediaBundle.Input(
                    "media_bundle", display_name="全媒体素材包",
                    tooltip="接「H3 素材与分段计划台」的全媒体素材包输出。",
                ),
                io.Combo.Input(
                    "task", display_name="任务类型", options=tasks, default=default_task,
                    tooltip="改写任务类型，例如 REF2AV 表示参考图生成音视频。",
                ),
                io.String.Input(
                    "prompt", display_name="待改写提示词", multiline=True, default="",
                    tooltip="写入原始提示词，改写器只围绕其中的媒体描述做优化。",
                ),
                io.Combo.Input(
                    "model", display_name="改写模型", options=models, default=models[0],
                    tooltip="提示词改写模型。列表为空表示未安装改写器后端。",
                ),
                io.Combo.Input(
                    "quantization", display_name="量化精度", options=quantizations, default=quantizations[0],
                    tooltip="改写模型的量化精度，显存吃紧时可选更低的档位。",
                ),
                io.Boolean.Input(
                    "greedy", display_name="贪心解码", default=True,
                    tooltip="开启为贪心解码；关闭则按设定的随机数采样。",
                ),
                io.Int.Input(
                    "seed", display_name="随机种子", default=42, min=0, max=0xFFFFFFFF, control_after_generate=True,
                ),
                io.Boolean.Input(
                    "keep_model_loaded", display_name="改写后保留模型", default=False,
                    tooltip="改写结束后不卸载模型，连续改写多条提示时可省去重复加载时间。",
                ),
                io.Int.Input(
                    "max_frames", display_name="最大帧数", default=8, min=1, max=64, optional=True,
                    tooltip="改写时允许参考的最大帧数；较小的值更快，但可能丢失长程动态。",
                ),
                io.Boolean.Input(
                    "bypass", display_name="直接透传", default=False, optional=True,
                    tooltip="开启后不调用改写器，直接把输入提示词原样输出，便于对比测试。",
                ),
            ],
            outputs=[io.String.Output(display_name="改写后提示词")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls,
        media_bundle,
        task,
        prompt,
        model,
        quantization,
        greedy,
        seed,
        keep_model_loaded,
        options=None,
        max_frames=8,
        bypass=False,
    ) -> io.NodeOutput:
        bundle = _require_prompt_media_bundle(media_bundle)
        if bypass:
            return io.NodeOutput((prompt or "").strip())
        if model == OMNI_MISSING_MODEL:
            raise RuntimeError(
                "请先安装 Prompt Rewriter Omni 后端：https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI"
            )

        module = _load_omni_rewriter()
        warning = _omni_compatibility_warning()
        if warning:
            log.warning(warning)
            print(f"[WJZ_H3_TimelineDirector] {warning}", flush=True)
        items = bundle["items"]
        try:
            max_references = int(module.MAX_REFERENCES)
        except (AttributeError, TypeError, ValueError):
            max_references = 12
        if len(items) > max_references:
            raise ValueError(
                f"The Omni media bundle contains {len(items)} references, but Prompt Rewriter Omni "
                f"supports at most {max_references}; remove unnecessary references from this segment."
            )
        supplied = {f"ref_{index}": item["value"] for index, item in enumerate(items)}
        reference_layout = json.dumps({"order": list(supplied)}, ensure_ascii=False)
        try:
            references, switched_off = module.arrange(supplied, reference_layout)
        except (AttributeError, TypeError, ValueError, KeyError) as exc:
            raise _omni_interface_error("arrange", exc) from exc
        if switched_off:
            raise RuntimeError("全媒体素材包里出现了意外禁用的参考项。")

        try:
            actual_kinds = [reference.kind for reference in references]
        except AttributeError as exc:
            raise _omni_interface_error("reference.kind", exc) from exc
        expected_kinds = [str(item.get("kind")) for item in items]
        if actual_kinds != expected_kinds:
            raise RuntimeError(
                f"全媒体顺序校验失败：期望 {expected_kinds}，实际得到 {actual_kinds}。"
            )
        try:
            settings = dict(module.DEFAULT_OPTIONS)
        except (AttributeError, TypeError) as exc:
            raise _omni_interface_error("DEFAULT_OPTIONS", exc) from exc
        if options:
            settings.update(options)
        node_id = getattr(getattr(cls, "hidden", None), "unique_id", None)
        try:
            progress = module.NodeProgress(node_id)
        except (AttributeError, TypeError) as exc:
            raise _omni_interface_error("NodeProgress", exc) from exc
        try:
            rewritten = _call_omni_rewrite(
                module,
                model=model,
                prompt=prompt,
                task=task,
                resolution=_closest_omni_resolution(bundle["width"], bundle["height"]),
                duration=float(bundle["generation_seconds"]),
                quantization=quantization,
                greedy=bool(greedy),
                seed=int(seed),
                keep_loaded=bool(keep_model_loaded),
                settings=settings,
                progress=progress,
                references=references,
                max_frames=int(max_frames),
            )
        except (AttributeError, TypeError, ValueError, KeyError) as exc:
            raise _omni_interface_error("rewrite_omni", exc) from exc
        progress.finish("提示词改写完成")
        return io.NodeOutput(rewritten)


class MiniMaxH3TimelineEncoder(io.ComfyNode):
    """Encode a planner result only after an external prompt rewrite completes."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WJZ_H3_TimelineEncoder",
            is_dev_only=True,
            display_name="H3 分段编码",
            description=(
                "把一份素材计划与最终 H3 提示词编码成参考图、原生 Guides、条件与音视频 latent。"
            ),
            category="model/conditioning/minimax",
            inputs=[
                io.Clip.Input(
                    "clip", display_name="源视频",
                    tooltip="H3 视频条件。所有分段的画面素材都从这条视频按时间线取帧。",
                ),
                io.Vae.Input("vae", display_name="视频 VAE", tooltip="负责画面 latent 的编解码。"),
                io.Vae.Input("audio_vae", display_name="音频 VAE", tooltip="负责音频 latent 的编解码。"),
                TimelinePlan.Input(
                    "plan", display_name="素材计划",
                    tooltip="接「H3 素材与分段计划台」的素材计划输出。",
                ),
                io.String.Input(
                    "prompt", display_name="分段提示词", multiline=True, dynamic_prompts=True,
                    tooltip="该分段的 H3 提示词，支持动态提示词插值。",
                ),
                io.Combo.Input(
                    "ref_image_size", display_name="参考图尺寸策略", options=["match", "max"], default="match",
                    tooltip="match：与输出分辨率一致；max：参考图允许放大到最大档，画质更好但更吃显存。",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="正向条件"),
                io.Latent.Output(display_name="音视频 latent", tooltip="分段 H3 画面与音频合并后的 latent。"),
                io.Audio.Output(
                    display_name="视频合并音轨",
                    tooltip="按时间线位置把裁剪后的源音频混进去，空隙处保留静音。",
                ),
                io.Audio.Output(
                    display_name="独立音轨",
                    tooltip="按素材箱顺序拼接独立的参考音频。",
                ),
            ],
        )

    @classmethod
    def execute(cls, clip, vae, audio_vae, plan, prompt, ref_image_size="match") -> io.NodeOutput:
        conditioning, latent, video_audio, standalone_audio = _encode_timeline_plan(
            plan, clip, vae, audio_vae, prompt, ref_image_size
        )
        return io.NodeOutput(conditioning, latent, video_audio, standalone_audio)


class MiniMaxH3TimelineDirector(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WJZ_H3_TimelineDirector",
            display_name="H3 导演台（一体化）",
            description=(
                "在可编辑的时间线上拼装 H3 参考素材。重叠的视频可选用原生 Add Guide、可编辑参考或仅边界模式；"
                "空隙区会自动锚定其边界帧。"
            ),
            category="model/conditioning/minimax",
            inputs=[
                io.Clip.Input(
                    "clip", display_name="源视频",
                    tooltip="H3 视频条件。所有参考素材都从这条视频按时间线取帧。",
                ),
                io.Vae.Input("vae", display_name="视频 VAE", tooltip="负责画面 latent 的编解码。"),
                io.Vae.Input("audio_vae", display_name="音频 VAE", tooltip="负责音频 latent 的编解码。"),
                io.String.Input(
                    "prompt", display_name="分段提示词", multiline=True, dynamic_prompts=True,
                    tooltip="该分段的 H3 提示词，支持动态提示词插值。",
                ),
                io.Int.Input(
                    "width", display_name="宽度", default=1344, min=32, max=16384, step=32,
                    tooltip="输出画面的像素宽度。",
                ),
                io.Int.Input(
                    "height", display_name="高度", default=768, min=32, max=16384, step=32,
                    tooltip="输出画面的像素高度。",
                ),
                io.Float.Input(
                    "generation_seconds", display_name="生成时长（秒）",
                    default=5.0, min=0.21, max=150.0, step=0.1,
                    tooltip="生成时长，与青色时间线区间双向同步。",
                ),
                io.Combo.Input(
                    "ref_image_size", display_name="参考图尺寸策略", options=["match", "max"], default="match",
                    tooltip="match：与输出分辨率一致；max：参考图允许放大到最大档，画质更好但更吃显存。",
                ),
                io.String.Input(
                    "timeline_data", display_name="时间线数据", default="", multiline=True,
                    tooltip="「H3 素材与分段计划台」导出后可粘回此处继续编辑。",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="正向条件"),
                io.Latent.Output(display_name="音视频 latent", tooltip="分段 H3 画面与音频合并后的 latent。"),
                io.Audio.Output(
                    display_name="视频合并音轨",
                    tooltip="按时间线位置把裁剪后的源音频混进去，空隙处保留静音。",
                ),
                io.Audio.Output(
                    display_name="独立音轨",
                    tooltip="按素材箱顺序拼接独立的参考音频。",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        vae,
        audio_vae,
        prompt,
        width,
        height,
        generation_seconds,
        ref_image_size="match",
        timeline_data="",
    ) -> io.NodeOutput:
        plan = _create_timeline_plan(timeline_data, width, height, generation_seconds)
        conditioning, latent, video_audio, standalone_audio = _encode_timeline_plan(
            plan, clip, vae, audio_vae, prompt, ref_image_size
        )
        return io.NodeOutput(conditioning, latent, video_audio, standalone_audio)
