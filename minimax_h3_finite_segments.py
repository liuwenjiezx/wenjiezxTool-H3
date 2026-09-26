"""本插件自有的有限分段 MiniMax H3 长视频规划与采样实现。"""

from __future__ import annotations

import copy
import json
import re
from functools import lru_cache
from pathlib import Path

import torch
import folder_paths
from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder

from .experimental_latent_guide import (
    _apply_linear_temporal_noise_mask,
    _valid_guide_frames,
)
from .drift_control_av import (
    drift_control_step_count,
    install_drift_control_av_model,
)
from .minimax_h3_timeline_director import (
    TimelinePlan,
    _require_timeline_plan,
    _timeline_for_prompt_index,
    _apply_h3_guides,
    _audio_mode,
    _decode_audio,
    _safe_input_path,
    _timeline_video_audio,
)

H3_FPS = 24
FiniteSegmentPlan = io.Custom("MINIMAX_H3_FINITE_SEGMENT_PLAN")


def _selflift_settings(
    step_count: int, requested_model: str = "", requested_high_steps=None,
) -> dict:
    if step_count < 2:
        raise ValueError("二阶采样至少需要两个采样步")
    if requested_high_steps is None:
        # Old saved workflows did not contain this field. Keep them usable while
        # making four full-resolution steps the default for the common 8-step run.
        high_steps = min(4, step_count - 1)
    else:
        try:
            high_steps = int(requested_high_steps)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"高清采样步数必须是整数；当前收到 {requested_high_steps!r}"
            ) from error
        if high_steps < 1 or high_steps >= step_count:
            raise ValueError(
                "高清采样步数至少为 1，且必须小于 Basic Scheduler 的步数"
                f"（当前为 {step_count}）；实际得到 {high_steps}"
            )
    transition = step_count - high_steps
    models = list(folder_paths.get_filename_list("latent_upscale_models"))
    if not models:
        raise ValueError(
            "二阶采样需要在 ComfyUI/models/latent_upscale_models 下提供一个 latent 超分模型"
        )
    selected = str(requested_model or "").strip() or models[0]
    if selected not in models:
        raise ValueError(
            f"所选的二阶 latent 超分模型不可用：{selected}"
        )
    return {
        "transition_step": transition, "lowres_scale": 0.5,
        "rho": 0.0, "w_min": 0.5, "w_max": 1.0,
        "upscaler_model": selected,
    }


@lru_cache(maxsize=8)
def _decode_locked_audio_file(
    path_text: str, modified_ns: int, file_size: int,
) -> dict | None:
    """把锁定音轨解码一次，并对每个分段复用它的 PCM 数据。

    ``modified_ns`` and ``file_size`` deliberately participate in the cache key,
    so replacing an upload at the same path cannot reuse stale audio.  Decoding
    from the beginning avoids compressed-audio seek/priming errors (notably the
    repeatable 1105-sample short read produced by some MP3 files).
    """

    del modified_ns, file_size
    return _decode_audio(Path(path_text), 0.0, None)


def _locked_audio_pcm(asset: dict) -> dict:
    path = _safe_input_path(str(asset["file"]))
    stat = path.stat()
    audio = _decode_locked_audio_file(
        str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size),
    )
    if audio is None:
        raise ValueError("锁定原始音频素材解码失败")
    return audio


def _video_soundtrack_lock_for_plan(plan) -> dict | None:
    """把启用中的参考视频音轨暴露为一条完整的时间线总轨。

    Reference-video generation must preserve the uploaded video's edited
    soundtrack in both one-stage and two-stage sampling, rather than merely
    offering it to H3 as a paired audio reference.  The synthetic asset keeps
    the existing locked-audio graph
    usable while ``_locked_audio_interval`` sources samples from the edited
    video timeline (including source trims, clip offsets, gaps, and mixes).
    """

    source = _require_timeline_plan(plan)
    timeline = source["timeline"]
    if timeline.get("videoAudioEnabled", True) is False:
        return None
    clips = [
        clip for clip in timeline.get("videoClips", [])
        if isinstance(clip, dict) and clip.get("file")
        and clip.get("hasAudio", True) is not False
        and float(clip.get("duration") or 0.0) > 0.0
    ]
    if not clips:
        return None
    identity = [
        {
            "file": str(clip["file"]),
            "start": round(float(clip.get("start") or 0.0), 6),
            "trimStart": round(float(clip.get("trimStart") or 0.0), 6),
            "duration": round(float(clip.get("duration") or 0.0), 6),
        }
        for clip in sorted(clips, key=lambda item: float(item.get("start") or 0.0))
    ]
    return {
        "lockKind": "timeline_video_audio",
        "name": "参考视频原始音轨",
        "identity": identity,
    }


def _locked_audio_for_plan(plan, *, include_video_soundtrack: bool = True) -> dict | None:
    source = _require_timeline_plan(plan)
    assets = [
        asset for asset in source["timeline"].get("audios", [])
        if isinstance(asset, dict) and asset.get("file") and _audio_mode(asset) == "locked"
    ]
    if len(assets) > 1:
        raise ValueError("每个分段最多只能有一个锁定原始音频素材")
    if assets:
        # An explicitly uploaded locked soundtrack always has priority over a
        # reference video's embedded audio.
        return assets[0]
    if include_video_soundtrack:
        return _video_soundtrack_lock_for_plan(source)
    return None


def _finite_locked_audio_asset(finite: dict) -> dict | None:
    """要求在整段渲染过程中使用一条连续的锁定音轨。"""

    plans = [
        _finite_plan_for_segment(finite, number)
        for number in range(1, int(finite["segment_count"]) + 1)
    ]
    assets = [
        _locked_audio_for_plan(plan, include_video_soundtrack=True)
        for plan in plans
    ]
    if not any(assets):
        return None
    if not all(assets):
        raise ValueError(
            "只有部分分段开启了锁定原始音频；请给每个分段使用同一份锁定音频"
        )
    identity = {
        json.dumps(asset.get("identity"), sort_keys=True, ensure_ascii=False)
        if asset.get("lockKind") == "timeline_video_audio"
        else json.dumps(
            [str(asset.get("file")), round(float(asset.get("trimStart") or 0.0), 6)],
            ensure_ascii=False,
        )
        for asset in assets if asset is not None
    }
    if len(identity) != 1:
        raise ValueError(
            "连续的锁定音轨必须在每个分段里使用同一音频文件与同一个源入点"
        )
    return copy.deepcopy(assets[0])


def _finite_video_audio_muted(finite: dict) -> bool:
    """当参考视频工作流明确关闭音频时返回 true。

    The video-audio switch is an output policy, not a request for H3 to invent
    a replacement soundtrack.  Explicitly uploaded locked audio still wins.
    """

    plans = [
        _finite_plan_for_segment(finite, number)
        for number in range(1, int(finite["segment_count"]) + 1)
    ]
    if any(
        _locked_audio_for_plan(plan, include_video_soundtrack=False) is not None
        for plan in plans
    ):
        return False
    has_reference_video = any(
        any(
            isinstance(clip, dict) and clip.get("file")
            for clip in _require_timeline_plan(plan)["timeline"].get("videoClips", [])
        )
        for plan in plans
    )
    if not has_reference_video:
        return False
    return all(
        _require_timeline_plan(plan)["timeline"].get("videoAudioEnabled", True) is False
        for plan in plans
    )


def _locked_audio_interval(plan, *, output_frames: int | None = None) -> dict:
    source = _require_timeline_plan(plan)
    asset = _locked_audio_for_plan(source)
    if asset is None:
        raise ValueError("该素材计划里没有锁定原始音频素材")
    selection = source["timeline"].get("selection") or {}
    timeline_start = max(0.0, float(selection.get("start") or 0.0))
    frame_count = int(output_frames or source.get("length") or 0)
    if frame_count < 1:
        raise ValueError("锁定音频区间没有目标帧数")
    duration = frame_count / H3_FPS
    if asset.get("lockKind") == "timeline_video_audio":
        # The mixed waveform is already laid out in timeline coordinates.
        source_start = timeline_start
        audio = _timeline_video_audio(source["timeline"])
    else:
        source_start = max(0.0, float(asset.get("trimStart") or 0.0)) + timeline_start
        audio = _locked_audio_pcm(asset)
    expected = round(duration * int(audio["sample_rate"]))
    source_sample = round(source_start * int(audio["sample_rate"]))
    waveform = audio["waveform"]
    source_sample = min(max(0, source_sample), int(waveform.shape[-1]))
    waveform = waveform[..., source_sample:source_sample + expected]
    if waveform.shape[-1] < expected:
        # H3 needs a legal AV latent duration even when the soundtrack ends in
        # the middle of a segment.  Silence is internal padding only: existing
        # source samples remain bit-for-bit unchanged, while the final output
        # follows the user's timeline duration without requiring frame-perfect
        # audio metadata.
        waveform = torch.nn.functional.pad(
            waveform, (0, expected - int(waveform.shape[-1])),
        )
    result = dict(audio)
    result["waveform"] = waveform[..., :expected].clone()
    return result


def _silent_audio_interval(plan, *, output_frames: int | None = None) -> dict:
    source = _require_timeline_plan(plan)
    frame_count = int(output_frames or source.get("length") or 0)
    if frame_count < 1:
        raise ValueError("静音音频区间没有目标帧数")
    sample_rate = 44100
    sample_count = max(1, round((frame_count / H3_FPS) * sample_rate))
    return {
        "waveform": torch.zeros((1, 1, sample_count), dtype=torch.float32),
        "sample_rate": sample_rate,
    }


def _parse_segment_prompts(value: str) -> list[str]:
    text = (value or "").strip()
    if not text:
        raise ValueError("分段提示词不能为空")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        parsed = parsed.get("segments")
    if isinstance(parsed, list):
        prompts = [str(item).strip() for item in parsed if str(item).strip()]
    else:
        prompts = [
            part.strip()
            for part in re.split(r"(?m)^\s*---\s*SEGMENT\s*---\s*$", text)
            if part.strip()
        ]
    if not prompts:
        raise ValueError("没有解析到任何分段提示词；请使用 JSON 数组或 --- SEGMENT --- 分隔符")
    return prompts


def _inject_continuity_instruction(prompt: str, overlap_frames: int) -> tuple[str, bool]:
    duration = overlap_frames / H3_FPS
    instruction = (
        f" The opening 00:00.000-00:{duration:06.3f} is a carried latent continuation "
        "from the preceding segment. Describe this opening as the preceding segment's "
        "final shot, preserving character positions, environment, motion, camera path, "
        "lighting, color, and sound before introducing new action."
    )
    for field in ("integrated_multimodal_description:", "detailed_description:"):
        field_index = prompt.find(field)
        if field_index < 0:
            continue
        shot_index = prompt.find("[Shot 1]", field_index + len(field))
        if shot_index >= 0:
            insert_at = shot_index + len("[Shot 1]")
            return prompt[:insert_at] + instruction + prompt[insert_at:], True
    return prompt, False


def _plan_for_segment(plan, segment_number: int):
    source = _require_timeline_plan(plan)
    if source.get("prompt_index") is not None:
        raise ValueError("有限分段需要完整的素材计划；请让「分段序号」保持悬空")
    timeline, selected, configured_count = _timeline_for_prompt_index(
        source["timeline"], segment_number
    )
    result = copy.deepcopy(source)
    result["timeline"] = timeline
    result["prompt_index"] = selected
    result["segment_count"] = configured_count
    return result


def _finite_plan_for_segment(finite: dict, segment_number: int):
    segment_plans = finite.get("segment_plans")
    if isinstance(segment_plans, list):
        index = int(segment_number) - 1
        if index < 0 or index >= len(segment_plans):
            raise ValueError(f"第 {segment_number} 个分段没有素材计划")
        return copy.deepcopy(_require_timeline_plan(segment_plans[index]))
    return _plan_for_segment(finite["source_plan"], segment_number)


def _prepare_finite_plan(
    plan,
    segment_prompts: str,
    segment_count: int,
    overlap_frames: int,
    inject_continuity: bool,
):
    source = _require_timeline_plan(plan)
    count = int(segment_count)
    configured_count = int(source.get("segment_count") or 0)
    if configured_count > 0 and configured_count != count:
        raise ValueError(
            f"The material planner has {configured_count} segments but finite expansion requests {count}; keep them identical"
        )
    prompts = _parse_segment_prompts(segment_prompts)
    if len(prompts) != count:
        raise ValueError(
            f"Finite expansion requests {count} segments but parsed {len(prompts)} prompts; keep them identical"
        )
    actual_overlap = _valid_guide_frames(int(overlap_frames))
    prepared = []
    for index, prompt in enumerate(prompts):
        if index > 0 and inject_continuity:
            prompt, injected = _inject_continuity_instruction(prompt, actual_overlap)
            if not injected:
                raise ValueError(
                    f"Segment {index + 1} has no [Shot 1] in the standard H3 fields; continuity instructions cannot be injected"
                )
        prepared.append(prompt)
    return {
        "type": "minimax_h3_finite_segment_plan",
        "version": 1,
        "source_plan": copy.deepcopy(source),
        "prompts": prepared,
        "segment_count": count,
        "requested_overlap_frames": int(overlap_frames),
        "overlap_frames": actual_overlap,
    }


def _prepare_timeline_segments(source):
    """编译计划台给出的帧窗口，但不改变它们的可见几何关系。"""
    source = _require_timeline_plan(source)
    if source.get("prompt_index") is not None:
        raise ValueError("生成全部时间线分段时请把「分段序号」保持悬空")
    config = source["timeline"].get("segmentConfig", {})
    segments = config.get("segments", [])
    count = int(config.get("count", 0))
    global_prompt = str(source["timeline"].get("globalPrompt") or "").strip()

    # The Material Planner is also the single-segment text-to-video planner.
    # With no windows, its current GEN selection becomes one finite segment;
    # the encoder created inside Finite Segment Sampling already knows how to
    # create an empty H3 AV latent when the plan contains no reference media.
    if count == 0:
        if not global_prompt:
            raise ValueError("请在素材计划台里填写全局提示词")
        length = int(source.get("length") or 0)
        if length < 5 or (length - 5) % 17 or length > 3592:
            raise ValueError("素材计划台的生成时长必须能换算成 5 + 17*n 帧")
        plan = copy.deepcopy(source)
        plan["timeline"]["segmentConfig"] = {"count": 0, "segments": []}
        return {
            "type": "minimax_h3_finite_segment_plan", "version": 4,
            "mode": "single_segment", "source_plan": source,
            "segment_count": 1, "segment_plans": [plan],
            "prompts": [global_prompt], "overlap_frames": 0,
            "segment_overlaps": [0], "segment_lengths": [length],
            "target_output_frames": length,
            "second_pass": bool(source["timeline"].get("secondPass")),
            "second_pass_model": str(source["timeline"].get("secondPassModel") or ""),
            "second_pass_high_steps": source["timeline"].get("secondPassHighSteps"),
        }

    if config.get("mode") != "timeline" or not 1 <= count <= 64:
        raise ValueError("请先点击素材计划台里的「更新分段」再生成")
    if len(segments) != count:
        raise ValueError("分段数量与时间线窗口数量不一致")
    local_prompts = [str(segment.get("prompt") or "").strip() for segment in segments]
    if any(local_prompts):
        if not all(local_prompts):
            missing = ", ".join(str(index + 1) for index, prompt in enumerate(local_prompts) if not prompt)
            raise ValueError(
                "Segment prompt mode is active because at least one segment has a prompt; "
                f"enter prompts for every segment (missing: {missing})"
            )
        resolved_prompts = local_prompts
    else:
        if not global_prompt:
            raise ValueError("请在素材计划台填写全局提示词，或为每个分段都写一条提示词")
        resolved_prompts = [global_prompt] * len(segments)

    plans, prompts, overlaps, lengths = [], [], [], []
    previous_start = previous_end = 0
    for index, segment in enumerate(segments):
        start, end = segment.get("startFrame"), segment.get("endFrame")
        if type(start) is not int or type(end) is not int:
            raise ValueError(f"第 {index + 1} 个分段必须使用整数帧边界")
        length = end - start
        if start < 0 or length < 5 or (length - 5) % 17 or length > 3592:
            raise ValueError(f"第 {index + 1} 个分段的长度必须是 5 + 17*n 帧（最长 150 秒）")
        overlap = previous_end - start if index else 0
        if (not index and start != 0) or (index and (
            start <= previous_start or end <= previous_end or overlap < 0
            or overlap >= min(length, lengths[-1])
            or (overlap and _valid_guide_frames(overlap) != overlap)
        )):
            raise ValueError(f"第 {index + 1} 个分段必须时序连续无空隙，并使用 H3 对齐的重叠帧数")
        plan = _plan_for_segment(source, index + 1)
        plan["timeline"]["selection"] = {"start": start / H3_FPS, "duration": length / H3_FPS}
        plan["generation_seconds"], plan["length"] = length / H3_FPS, length
        plan["timeline"]["segmentConfig"] = {"count": 0, "segments": []}
        # Selected audio is local to this segment (short timbre references remain reusable).
        plans.append(plan)
        prompts.append(resolved_prompts[index])
        overlaps.append(overlap)
        lengths.append(length)
        previous_start, previous_end = start, end
    return {
        "type": "minimax_h3_finite_segment_plan", "version": 3,
        "mode": "timeline_segments", "source_plan": source,
        "segment_count": len(plans), "segment_plans": plans, "prompts": prompts,
        "overlap_frames": 0, "segment_overlaps": overlaps,
        "segment_lengths": lengths, "target_output_frames": previous_end,
        "second_pass": bool(source["timeline"].get("secondPass")),
        "second_pass_model": str(source["timeline"].get("secondPassModel") or ""),
        "second_pass_high_steps": source["timeline"].get("secondPassHighSteps"),
    }


def _require_finite_plan(value):
    if isinstance(value, dict) and value.get("type") == "MINIMAX_H3_TIMELINE_PLAN":
        value = _prepare_timeline_segments(value)
    if not isinstance(value, dict) or value.get("type") != "minimax_h3_finite_segment_plan":
        raise ValueError("finite_plan 必须来自 MiniMax H3 素材计划台或旧的有限分段计划")
    count = int(value.get("segment_count") or 0)
    prompts = value.get("prompts")
    if count < 1 or not isinstance(prompts, list) or len(prompts) != count:
        raise ValueError("有限分段计划不完整；请更新分段计划后重新运行")
    segment_plans = value.get("segment_plans")
    if segment_plans is not None and (
        not isinstance(segment_plans, list) or len(segment_plans) != count
    ):
        raise ValueError("长参考计划中某些分段的素材不完整")
    return value


class MiniMaxH3FiniteSegmentExpansion(io.ComfyNode):
    """校验提示词与素材分配，产出一份可复用的有限分段计划。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FiniteSegmentExpansion",
            is_deprecated=True,
            display_name="H3 分段扩展（已停用）",
            category="MiniMax H3/Long Video",
            description=(
                "解析分段提示词、校验分段数量、匹配各分段的素材，最终生成一份有限分段计划。"
                "本节点不加载模型、不做调度、也不采样。"
            ),
            inputs=[
                TimelinePlan.Input("plan", display_name="素材计划"),
                io.String.Input(
                    "segment_prompts", display_name="分段提示词", multiline=True,
                    tooltip="每段一条提示词；留空则回退到全局提示词。用 --- SEGMENT --- 或 JSON 数组分隔多条。",
                ),
                io.Int.Input("segment_count", display_name="分段数", default=3, min=1, max=12),
                io.Int.Input(
                    "overlap_frames", display_name="重叠帧数", default=22,
                    min=1, max=362, tooltip="会向下取整到合法的 1 或 5/22/39/56… 帧数。",
                ),
                io.Boolean.Input(
                    "inject_continuity_instruction", display_name="注入开场连贯性", default=True,
                    tooltip="开启后会自动在提示词前追加一段开场描述，说明与上一分段末镜头的连贯关系。",
                ),
            ],
            outputs=[
                FiniteSegmentPlan.Output(display_name="分段计划"),
                io.Int.Output(display_name="实际重叠帧数"),
                io.String.Output(display_name="计划状态"),
            ],
        )

    @classmethod
    def execute(
        cls, plan, segment_prompts, segment_count, overlap_frames,
        inject_continuity_instruction,
    ):
        finite = _prepare_finite_plan(
            plan, segment_prompts, segment_count, overlap_frames,
            bool(inject_continuity_instruction),
        )
        overlap = finite["overlap_frames"]
        status = (
            f"已规划 {finite['segment_count']} 个分段；实际重叠 {overlap} 帧 "
            f"（{overlap / H3_FPS:.3f}s）。本节点不进行采样。"
        )
        return io.NodeOutput(finite, overlap, status)


class MiniMaxH3FiniteLatentContinuation(io.ComfyNode):
    """内部有限图节点：携带上一段的音视频 latent 尾部。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WJZ_H3_FiniteLatentContinuation",
            display_name="H3 分段接续（内部）",
            category="MiniMax H3/Internal",
            is_dev_only=True,
            inputs=[
                io.Conditioning.Input("positive", display_name="正向条件"),
                io.Latent.Input("target_latent", display_name="目标 latent"),
                io.Int.Input("iteration", display_name="序号", force_input=True,
                             tooltip="当前分段的序号，从 0 开始；0 表示第一段。"),
                io.Int.Input(
                    "overlap_frames", display_name="重叠帧数", default=22, min=0, max=3592,
                    tooltip="与上一个分段的重叠帧数，决定续写范围。",
                ),
                io.Boolean.Input(
                    "continue_audio_latent", display_name="接续音频 latent", default=True,
                    tooltip="开启时上一段的音频 latent 会作为噪声遮罩的一部分参与续写。",
                ),
                io.Model.Input("model", display_name="采样模型"),
                io.Sigmas.Input("sigmas", display_name="sigma 调度"),
                io.Latent.Input("previous_latent", display_name="上一段 latent", optional=True),
                io.Image.Input("previous_images", display_name="上一段末帧", optional=True),
                io.Vae.Input("vae", display_name="视频 VAE", optional=True),
                io.Vae.Input("audio_vae", display_name="音频 VAE", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="正向条件"),
                io.Latent.Output(display_name="目标 latent"),
                io.Int.Output(display_name="实际重叠帧数"),
                io.Model.Output(display_name="采样模型"),
            ],
        )

    @classmethod
    def execute(
        cls, positive, target_latent, iteration, overlap_frames,
        continue_audio_latent, model, sigmas, previous_latent=None,
        previous_images=None, vae=None, audio_vae=None,
    ):
        if int(overlap_frames) == 0:
            # Touching windows are independent: no video or audio continuation.
            return io.NodeOutput(positive, target_latent, 0, model)
        if int(iteration) > 0 and int(overlap_frames) == 1:
            if previous_images is None or vae is None or audio_vae is None:
                raise ValueError("相邻分段需要提供前一个分段的末帧图片与对应的 VAE")
            positive = _apply_h3_guides(positive, target_latent, vae, audio_vae, [{
                "image": previous_images[-1:].clone(), "audio": None, "frame_idx": 0,
            }])
            return io.NodeOutput(positive, target_latent, int(overlap_frames), model)
        actual_overlap = _valid_guide_frames(int(overlap_frames))
        if int(iteration) <= 0:
            return io.NodeOutput(positive, target_latent, 0 if int(overlap_frames) == 0 else actual_overlap, model)
        if previous_latent is None:
            raise ValueError("第 2 段及以后的分段需要提供上一段采样出的 latent")
        masked_target, details = _apply_linear_temporal_noise_mask(
            target_latent=target_latent,
            source_latent=previous_latent,
            guide_frames=actual_overlap,
            include_audio=bool(continue_audio_latent),
            gradient=False,
            audio_soft_release=bool(continue_audio_latent),
        )
        # SelfLift keeps its native pre-lift low-resolution prediction beside
        # the final high-resolution result. Carry that low-grid state into the
        # next segment so Drift-Control does not reconstruct it by shrinking
        # the preceding final high-resolution latent again.
        previous_low_carry = previous_latent.get("selflift_low_resolution_carry")
        if previous_low_carry is not None:
            masked_target = dict(masked_target)
            masked_target["selflift_previous_low_resolution_carry"] = previous_low_carry
        patched_model = install_drift_control_av_model(
            model, masked_target, sigmas, prefix_steps=details["video_tokens"]
        )
        return io.NodeOutput(positive, masked_target, details["frames"], patched_model)


class MiniMaxH3LockedAudioSlice(io.ComfyNode):
    """内部节点：读取某个 GEN 窗口精确对应的音轨区间。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WJZ_H3_LockedAudioSlice",
            display_name="H3 锁定音轨切片（内部）",
            category="MiniMax H3/Internal",
            is_dev_only=True,
            inputs=[TimelinePlan.Input("plan", display_name="素材计划")],
            outputs=[io.Audio.Output(display_name="锁定音轨")],
        )

    @classmethod
    def execute(cls, plan):
        return io.NodeOutput(_locked_audio_interval(plan))


class MiniMaxH3SilentAudioSlice(io.ComfyNode):
    """内部节点：把某个参考视频分段固定为静音。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WJZ_H3_SilentAudioSlice",
            display_name="H3 静音切片（内部）",
            category="MiniMax H3/Internal",
            is_dev_only=True,
            inputs=[TimelinePlan.Input("plan", display_name="素材计划")],
            outputs=[io.Audio.Output(display_name="静音音轨")],
        )

    @classmethod
    def execute(cls, plan):
        return io.NodeOutput(_silent_audio_interval(plan))


class MiniMaxH3LockAudioLatent(io.ComfyNode):
    """替换 H3 目标音频流，并把该流排除在去噪之外。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WJZ_H3_LockAudioLatent",
            display_name="H3 锁定音频 latent（内部）",
            category="MiniMax H3/Internal",
            is_dev_only=True,
            inputs=[
                io.Latent.Input("target_latent", display_name="目标 latent",
                                tooltip="待写入的音视频 latent。"),
                io.Latent.Input("audio_latent", display_name="音频 latent",
                                tooltip="已编码的音频 latent，会覆盖目标 latet 中的音频流。"),
            ],
            outputs=[io.Latent.Output(display_name="锁定 AV latent",
                                      tooltip="音频流已锁定、并被排除在去噪之外的 latent。")],
        )

    @classmethod
    def execute(cls, target_latent, audio_latent):
        target_samples = target_latent.get("samples") if isinstance(target_latent, dict) else None
        source_audio = audio_latent.get("samples") if isinstance(audio_latent, dict) else None
        if target_samples is None or not getattr(target_samples, "is_nested", False):
            raise ValueError("锁定音频需要一份嵌套的 MiniMax H3 音视频 latent")
        streams = list(target_samples.unbind())
        if len(streams) != 2 or source_audio is None or getattr(source_audio, "is_nested", False):
            raise ValueError("锁定音频需要一份已编码的音频 latent")
        video, target_audio = streams
        source_audio = source_audio.to(device=target_audio.device, dtype=target_audio.dtype)
        if source_audio.ndim != target_audio.ndim or tuple(source_audio.shape[:-1]) != tuple(target_audio.shape[:-1]):
            raise ValueError(
                f"已编码的锁定音频 {tuple(source_audio.shape)} 与 H3 目标音频 {tuple(target_audio.shape)} 不匹配"
            )
        if source_audio.shape[-1] < target_audio.shape[-1]:
            source_audio = torch.nn.functional.pad(
                source_audio, (0, int(target_audio.shape[-1] - source_audio.shape[-1]))
            )
        locked_audio = source_audio[..., :target_audio.shape[-1]].clone()

        existing_mask = target_latent.get("noise_mask")
        if existing_mask is not None and getattr(existing_mask, "is_nested", False):
            video_mask = existing_mask.unbind()[0]
        elif torch.is_tensor(existing_mask):
            video_mask = existing_mask
        else:
            video_mask = torch.ones(
                (video.shape[0], 1, video.shape[2], 1, 1),
                dtype=torch.float32, device=video.device,
            )
        audio_mask = torch.zeros(
            (target_audio.shape[0], 1, 1, target_audio.shape[-1]),
            dtype=torch.float32, device=target_audio.device,
        )
        output = dict(target_latent)
        output["samples"] = NestedTensor((video, locked_audio))
        output["noise_mask"] = NestedTensor((video_mask, audio_mask))
        return io.NodeOutput(output)


class MiniMaxH3LockedAudioMaster(io.ComfyNode):
    """只解码一次连续的原始波形，供最终视频输出使用。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WJZ_H3_LockedAudioMaster",
            display_name="H3 锁定音轨总轨（内部）",
            category="MiniMax H3/Internal",
            is_dev_only=True,
            inputs=[FiniteSegmentPlan.Input("finite_plan", display_name="分段计划")],
            outputs=[io.Audio.Output(display_name="原始音轨")],
        )

    @classmethod
    def execute(cls, finite_plan):
        finite = _require_finite_plan(finite_plan)
        asset = _finite_locked_audio_asset(finite)
        if asset is None:
            raise ValueError("有限分段计划里没有锁定的原始音轨")
        first = _finite_plan_for_segment(finite, 1)
        target_frames = int(finite.get("target_output_frames") or 0)
        if target_frames < 1:
            target_frames = sum(int(value) for value in finite.get("segment_lengths", []))
            target_frames -= sum(int(value) for value in finite.get("segment_overlaps", [])[1:])
        if target_frames < 1:
            target_frames = int(first.get("length") or 0)
        master_plan = copy.deepcopy(first)
        if asset.get("lockKind") != "timeline_video_audio":
            master_plan["timeline"]["audios"] = [asset]
        return io.NodeOutput(_locked_audio_interval(master_plan, output_frames=target_frames))


class MiniMaxH3SilentAudioMaster(io.ComfyNode):
    """当视频音频被关闭时，返回一条时长精确对齐的静音总轨。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WJZ_H3_SilentAudioMaster",
            display_name="H3 静音总轨（内部）",
            category="MiniMax H3/Internal",
            is_dev_only=True,
            inputs=[FiniteSegmentPlan.Input("finite_plan", display_name="分段计划")],
            outputs=[io.Audio.Output(display_name="静音音轨")],
        )

    @classmethod
    def execute(cls, finite_plan):
        finite = _require_finite_plan(finite_plan)
        first = _finite_plan_for_segment(finite, 1)
        target_frames = int(finite.get("target_output_frames") or 0)
        if target_frames < 1:
            target_frames = sum(int(value) for value in finite.get("segment_lengths", []))
            target_frames -= sum(int(value) for value in finite.get("segment_overlaps", [])[1:])
        if target_frames < 1:
            target_frames = int(first.get("length") or 0)
        return io.NodeOutput(_silent_audio_interval(first, output_frames=target_frames))


class MiniMaxH3FiniteSegmentFinalize(io.ComfyNode):
    """内部有限图节点：移除已解码的重叠部分。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WJZ_H3_FiniteSegmentFinalize",
            display_name="H3 分段收尾（内部）",
            category="MiniMax H3/Internal",
            is_dev_only=True,
            inputs=[
                io.Latent.Input("sampled_latent", display_name="采样 latent"),
                io.Image.Input("images", display_name="本段解码帧"),
                io.Int.Input("iteration", display_name="序号", force_input=True),
                io.Int.Input("overlap_frames", display_name="重叠帧数", default=22, min=0, max=3592),
                io.Boolean.Input(
                    "trim_audio_head", display_name="裁掉音频重叠头", default=True,
                    tooltip="开启时移除本段音频开头与重叠部分重合的采样，避免与上一段重复。",
                ),
                io.Audio.Input("audio", display_name="本段音频", optional=True),
                io.Image.Input("accumulated_images", display_name="已累积帧", optional=True),
            ],
            outputs=[
                io.Latent.Output(display_name="完整 latent", tooltip="去掉重叠后的完整 latent。"),
                io.Image.Output(display_name="去重帧", tooltip="已去掉重叠重复区间的画面帧。"),
                io.Audio.Output(display_name="去重音频", tooltip="已去掉重叠重复区间的音频。"),
            ],
        )

    @classmethod
    def execute(
        cls, sampled_latent, images, iteration, overlap_frames,
        trim_audio_head=True, audio=None, accumulated_images=None,
    ):
        trim_frames = 0 if int(iteration) <= 0 or int(overlap_frames) == 0 else _valid_guide_frames(int(overlap_frames))
        if images.shape[0] <= trim_frames:
            raise ValueError(f"本分段只有 {images.shape[0]} 帧；无法再裁掉 {trim_frames} 帧的重叠")
        trimmed_images = images[trim_frames:].clone() if trim_frames else images
        if accumulated_images is not None:
            # The preceding segment owns the visual overlap. Preserve its
            # decoded tail and discard the duplicate opening interval from the
            # incoming segment before joining the two timelines.
            trimmed_images = torch.cat((accumulated_images, trimmed_images), dim=0)
        trimmed_audio = audio
        if audio is not None and bool(trim_audio_head):
            waveform = audio.get("waveform")
            sample_rate = int(audio.get("sample_rate", 0))
            if waveform is None or sample_rate <= 0:
                raise ValueError("audio 必须包含 waveform 字段与合法的 sample_rate")
            trim_samples = round((trim_frames / H3_FPS) * sample_rate)
            if waveform.shape[-1] <= trim_samples:
                raise ValueError("本分段的音频太短，无法裁掉重叠部分")
            trimmed_audio = dict(audio)
            trimmed_audio["waveform"] = (
                waveform[..., trim_samples:].clone() if trim_samples else waveform
            )
        return io.NodeOutput(sampled_latent, trimmed_images, trimmed_audio)


class MiniMaxH3FiniteAudioTrimTail(io.ComfyNode):
    """内部节点：让进入的 Soft AV 分段掌握接缝的所有权。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WJZ_H3_FiniteAudioTrimTail",
            display_name="H3 尾部音频裁切（内部）",
            category="MiniMax H3/Internal",
            is_dev_only=True,
            inputs=[
                io.Audio.Input("audio", display_name="待裁切音频"),
                io.Int.Input(
                    "overlap_frames", display_name="重叠帧数", default=39, min=1, max=3592,
                    tooltip="要从音频尾部裁掉的重叠帧数（Soft AV 半余弦释放区）。",
                ),
            ],
            outputs=[io.Audio.Output(display_name="裁切后音频",
                                     tooltip="尾部重叠样本已裁掉的音频。")],
        )

    @classmethod
    def execute(cls, audio, overlap_frames):
        waveform = audio.get("waveform") if isinstance(audio, dict) else None
        sample_rate = int(audio.get("sample_rate", 0)) if isinstance(audio, dict) else 0
        if waveform is None or sample_rate <= 0:
            raise ValueError("audio 必须包含 waveform 字段与合法的 sample_rate")
        trim_samples = round((_valid_guide_frames(int(overlap_frames)) / H3_FPS) * sample_rate)
        if waveform.shape[-1] <= trim_samples:
            raise ValueError("累积的音频太短，无法替换重叠尾部")
        output = dict(audio)
        output["waveform"] = waveform[..., :-trim_samples].clone()
        return io.NodeOutput(output)


class MiniMaxH3FiniteOutputTrim(io.ComfyNode):
    """把自动分段的补帧裁回到最长源素材的时长。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WJZ_H3_FiniteOutputTrim",
            display_name="H3 成片裁切（内部）",
            category="MiniMax H3/Internal",
            is_dev_only=True,
            inputs=[
                io.Image.Input("images", display_name="合并帧"),
                io.Audio.Input("audio", display_name="合并音频"),
                io.Int.Input(
                    "output_frames", display_name="目标帧数", default=5, min=1, force_input=True,
                    tooltip="最终成片的帧数；多余的尾部帧与对应音频采样会被裁掉。",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="裁切后帧", tooltip="按目标帧数裁好的画面帧。"),
                io.Audio.Output(display_name="裁切后音频",
                                 tooltip="与目标帧数时长对齐的音频。"),
            ],
        )

    @classmethod
    def execute(cls, images, audio, output_frames):
        frame_count = int(output_frames)
        if int(images.shape[0]) < frame_count:
            raise ValueError(
                f"生成的输出只有 {images.shape[0]} 帧，无法还原成 {frame_count} 帧的源素材时长"
            )
        trimmed_images = images[:frame_count].clone()
        trimmed_audio = audio
        waveform = audio.get("waveform") if isinstance(audio, dict) else None
        sample_rate = int(audio.get("sample_rate", 0)) if isinstance(audio, dict) else 0
        if waveform is not None and sample_rate > 0:
            output_samples = round((frame_count / H3_FPS) * sample_rate)
            if waveform.shape[-1] > output_samples:
                trimmed_audio = dict(audio)
                trimmed_audio["waveform"] = waveform[..., :output_samples].clone()
        return io.NodeOutput(trimmed_images, trimmed_audio)


class MiniMaxH3FiniteSegmentSampler(io.ComfyNode):
    """把有限分段计划展开成标准的无环采样图。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WJZ_H3_FiniteSegmentSampler",
            display_name="H3 分段采样（长视频）",
            category="MiniMax H3/Long Video",
            description=(
                "把有限分段计划展开成标准的无环采样图。采样器与调度器仍然是外部节点，"
                "不需要 Loop / Loop Variable / Close Loop 等节点。"
            ),
            enable_expand=True,
            inputs=[
                io.Model.Input("model", display_name="采样模型",
                                tooltip="H3 模型，所有分段共用。"),
                io.Clip.Input("clip", display_name="源视频", tooltip="H3 视频条件。"),
                io.Vae.Input("vae", display_name="视频 VAE", tooltip="负责画面 latent 的编解码。"),
                io.Vae.Input("audio_vae", display_name="音频 VAE",
                                tooltip="负责音频 latent 的编解码。"),
                FiniteSegmentPlan.Input("finite_plan", display_name="分段计划",
                                        tooltip="接「H3 素材与分段计划台」导出的分段计划。"),
                io.Sampler.Input("sampler", display_name="采样器"),
                io.Sigmas.Input("sigmas", display_name="sigma 调度"),
                io.Int.Input("seed", display_name="随机种子", default=0, min=0,
                             max=0xFFFFFFFFFFFFFFFF, control_after_generate=True,
                             tooltip="所有分段共用同一个种子。"),
                io.Boolean.Input(
                    "continue_audio_latent", display_name="接续音频 latent", default=True,
                    tooltip="关闭后每段音频独立生成，不从上一段继承。",
                ),
                io.Combo.Input(
                    "ref_image_size", display_name="参考图尺寸策略",
                    options=["match", "max"], default="match",
                    tooltip="match：与输出分辨率一致；max：参考图可放大到最大档，更吃显存。",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="末段 latent",
                                 tooltip="最后一段采样出的 latent，可继续接后续处理。"),
                io.Image.Output(display_name="合并帧", tooltip="所有分段拼接、去重后的画面帧。"),
                io.Audio.Output(display_name="合并音频", tooltip="所有分段拼接、去重后的音频。"),
                io.String.Output(display_name="采样状态",
                                 tooltip="本次展开与采样的模式说明，便于核对音轨处理方式。"),
            ],
        )

    @classmethod
    def execute(
        cls, model, clip, vae, audio_vae, finite_plan, sampler, sigmas, seed,
        continue_audio_latent, ref_image_size="match",
    ):
        finite = _require_finite_plan(finite_plan)
        graph = GraphBuilder()
        previous_latent = None
        previous_images = None
        merged_images = None
        merged_audio = None
        last_sampled = None
        overlap = int(finite["overlap_frames"])
        locked_audio = _finite_locked_audio_asset(finite)
        muted_video_audio = _finite_video_audio_muted(finite)
        fixed_audio = locked_audio is not None or muted_video_audio
        soft_audio = bool(continue_audio_latent) and not fixed_audio
        steps = drift_control_step_count(sigmas)
        if steps < 1:
            raise ValueError(
                "Drift-Control AV 需要一条至少包含一个采样步的 sigma 调度"
            )
        second_pass = bool(finite.get("second_pass"))
        selflift_settings = (
            _selflift_settings(
                steps,
                finite.get("second_pass_model", ""),
                finite.get("second_pass_high_steps"),
            )
            if second_pass else None
        )

        for index, prompt in enumerate(finite["prompts"]):
            number = index + 1
            overlap = int(finite.get("segment_overlaps", [overlap] * finite["segment_count"])[index])
            segment_plan = _finite_plan_for_segment(finite, number)
            encoder = graph.node(
                "WJZ_H3_TimelineEncoder", id=f"encode_{number}",
                clip=clip, vae=vae, audio_vae=audio_vae,
                plan=segment_plan,
                prompt=prompt, ref_image_size=ref_image_size,
            )
            continuation_inputs = {
                "positive": encoder.out(0), "target_latent": encoder.out(1),
                "iteration": index, "overlap_frames": overlap,
                "continue_audio_latent": bool(continue_audio_latent) and not fixed_audio,
                "model": model, "sigmas": sigmas,
            }
            if previous_latent is not None and overlap > 0:
                continuation_inputs["previous_latent"] = previous_latent
                if overlap == 1:
                    continuation_inputs.update(previous_images=previous_images, vae=vae, audio_vae=audio_vae)
            continuation = graph.node(
                "WJZ_H3_FiniteLatentContinuation", id=f"continue_{number}",
                **continuation_inputs,
            )
            sampling_latent = continuation.out(1)
            if fixed_audio:
                source_audio = graph.node(
                    "WJZ_H3_LockedAudioSlice" if locked_audio is not None else "WJZ_H3_SilentAudioSlice",
                    id=(f"locked_audio_slice_{number}" if locked_audio is not None else f"silent_audio_slice_{number}"),
                    plan=segment_plan,
                )
                encoded_audio = graph.node(
                    "VAEEncodeAudio", id=f"fixed_audio_encode_{number}",
                    audio=source_audio.out(0), vae=audio_vae,
                )
                sampling_latent = graph.node(
                    "WJZ_H3_LockAudioLatent", id=f"fixed_audio_latent_{number}",
                    target_latent=sampling_latent, audio_latent=encoded_audio.out(0),
                ).out(0)
            if second_pass:
                sampled = graph.node(
                    "WJZ_H3_TimelineSelfLiftSampler", id=f"sample_{number}",
                    model=continuation.out(3), positive=continuation.out(0),
                    negative=continuation.out(0), vae=vae,
                    latent_image=sampling_latent, sampler=sampler, sigmas=sigmas,
                    seed=int(seed), cfg=1.0, **selflift_settings,
                )
            else:
                noise = graph.node("RandomNoise", id=f"noise_{number}", noise_seed=int(seed))
                guider = graph.node(
                    "BasicGuider", id=f"guider_{number}", model=continuation.out(3),
                    conditioning=continuation.out(0),
                )
                sampled = graph.node(
                    "SamplerCustomAdvanced", id=f"sample_{number}", noise=noise.out(0),
                    guider=guider.out(0), sampler=sampler, sigmas=sigmas,
                    latent_image=sampling_latent,
                )
            images = graph.node(
                "VAEDecode", id=f"decode_video_{number}", samples=sampled.out(0), vae=vae,
            )
            audio = graph.node(
                "VAEDecodeAudio", id=f"decode_audio_{number}", samples=sampled.out(0), vae=audio_vae,
            )
            finalize_inputs = {}
            if merged_images is not None:
                finalize_inputs["accumulated_images"] = merged_images
            finalized = graph.node(
                "WJZ_H3_FiniteSegmentFinalize", id=f"finalize_{number}",
                sampled_latent=sampled.out(0), images=images.out(0), audio=audio.out(0),
                iteration=index, overlap_frames=overlap,
                trim_audio_head=not soft_audio,
                **finalize_inputs,
            )
            current_images, current_audio = finalized.out(1), finalized.out(2)
            if merged_images is None:
                merged_images, merged_audio = current_images, current_audio
            else:
                previous_audio_for_join = merged_audio
                if soft_audio and overlap > 0:
                    previous_audio_for_join = graph.node(
                        "WJZ_H3_FiniteAudioTrimTail", id=f"trim_audio_tail_{number}",
                        audio=merged_audio, overlap_frames=overlap,
                    ).out(0)
                audio_join = graph.node(
                    "AudioConcat", id=f"join_audio_{number}",
                    audio1=previous_audio_for_join, audio2=current_audio, direction="after",
                )
                merged_images, merged_audio = current_images, audio_join.out(0)
            previous_latent = sampled.out(0)
            previous_images = images.out(0)
            last_sampled = sampled.out(0)

        target_output_frames = int(finite.get("target_output_frames") or 0)
        if target_output_frames > 0:
            output_trim = graph.node(
                "WJZ_H3_FiniteOutputTrim", id="trim_auto_segment_output",
                images=merged_images, audio=merged_audio,
                output_frames=target_output_frames,
            )
            merged_images, merged_audio = output_trim.out(0), output_trim.out(1)

        if locked_audio is not None:
            merged_audio = graph.node(
                "WJZ_H3_LockedAudioMaster", id="locked_audio_master",
                finite_plan=finite,
            ).out(0)
        elif muted_video_audio:
            merged_audio = graph.node(
                "WJZ_H3_SilentAudioMaster", id="silent_audio_master",
                finite_plan=finite,
            ).out(0)

        locked_video_soundtrack = bool(
            locked_audio is not None
            and locked_audio.get("lockKind") == "timeline_video_audio"
        )
        mode_status = (
            "参考视频音轨以零音频去噪掩码编码进每个分段，最终输出使用该时间线精确编辑后的波形"
            if locked_video_soundtrack else
            "源音轨以零音频去噪掩码编码进每个分段，最终输出使用原始连续波形"
            if locked_audio is not None else
            "参考视频音频已关闭，每个分段都使用零去噪的静音 latent，最终输出为静音"
            if muted_video_audio else
            f"Drift-Control AV：{overlap} 帧遮罩适配了 {steps} 个采样步；重叠音频使用 8 拍 Soft AV 半余弦释放"
            if continue_audio_latent
            else f"Drift-Control AV：{overlap} 帧遮罩适配了 {steps} 个采样步；音频独立生成"
        )
        status = (
            f"已展开并采样 {finite['segment_count']} 个分段；实际重叠 {overlap} 帧；"
            f"所有分段共用种子 {int(seed)}；{mode_status}；"
            f"音频 latent {'锁定到源音轨' if locked_audio is not None else ('固定为静音' if muted_video_audio else ('接续' if continue_audio_latent else '不接续'))}。"
        )
        if finite.get("mode") == "timeline_segments":
            status = (
                f"已采样 {finite['segment_count']} 个时间线窗口；各段长度={finite['segment_lengths']}；"
                f"接缝重叠={finite['segment_overlaps']}。零重叠接缝各自独立生成，不参考上一段、也不裁帧。"
            )
            if locked_audio is not None:
                status += (
                    " 参考视频音轨" if locked_video_soundtrack
                    else " 上传的锁定音轨"
                )
                status += (
                    " 以零音频去噪掩码编码进每个分段；最终音频为原始连续波形。"
                )
            elif muted_video_audio:
                status += (
                    " 视频源音频已关闭，每个分段都使用零去噪的静音 latent，最终输出为静音。"
                )
        if second_pass:
            status += (
                f"每个分段都进行了二阶采样：{selflift_settings['transition_step']} 个低清步 + "
                f"{steps - selflift_settings['transition_step']} 个全清步；第 2 段起会复用上一段的 "
                "原生低清尾部作为低清阶段起点，并以上一段的最终全清尾部作为高清阶段的掩码开头锚点。"
            )
        if target_output_frames > 0:
            trim_tail_frames = int(finite.get("trim_tail_frames") or 0)
            status += (
                f"已移除末尾多出的 {trim_tail_frames} 帧及其对应音频采样；输出时长与源素材完全一致（{target_output_frames} 帧）。"
            )
        return io.NodeOutput(
            last_sampled, merged_images, merged_audio, status, expand=graph.finalize()
        )
