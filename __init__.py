# -*- coding: utf-8 -*-
"""wenjiezxTool-H3 —— MiniMax H3 导演台节点包（**开源 · GPL-3.0**）

这个包是做什么的
----------------
把「MiniMax H3 视频生成」用到的导演台节点收在一处，方便 ComfyUI 服务器助手
在给用户装环境时**一次性装上**，不用再让用户自己去 ComfyUI-Manager 里翻。

节点类名保持原样，但**注册键（NODE_CLASS_MAPPINGS 的 key）已从 MiniMaxH3*
改为 WJZ_H3_***：原键与上游原版插件（ComfyUI-MiniMaxH3-TimelineDirector）
完全相同，用户如果同时装了两份，会触发 ComfyUI「节点重复注册」冲突，谁后加载
谁覆盖，行为不可预测。改成 WJZ_ 前缀后两套节点可共存互不干扰。

注意：旧工作流 JSON 里存的还是旧键名（MiniMaxH3*），打开时会提示缺节点——
如果机器上装着原版插件，会自动落到原版节点上继续跑；要彻底换新键需批量迁移
工作流 JSON。
菜单里的显示名已全部换成中文（含 schema 内写死的 display_name、输出口/输入口标签）。

许可：GPL-3.0（与上游一致）
--------------------------
本包**基于开源项目** `ComfyUI-MiniMaxH3-TimelineDirector`（作者 Songssx，
GPL-3.0）整理而成。GPL-3.0 允许自由使用、修改与再分发，条件是衍生作品
**同样以 GPL-3.0 开源** —— 所以这个包是开源的，源码就在这里，没有做闭源编译。

这也是它和 `wenjiezxTool`（闭源 .pyd）分开成两个包的原因：把 GPL 的代码
塞进闭源插件会违反许可，分开就各自合规。

* 上游：`https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector`
* 本包改动：整理目录、中文显示名、中文说明文档；节点逻辑未作功能性修改。
* 完整许可文本见 `LICENSE-GPL.txt`。
"""

from .minimax_h3_timeline_director import (
    MiniMaxH3TimelineDirector,
    MiniMaxH3TimelineEncoder,
    MiniMaxH3OmniPromptBridge,
    MiniMaxH3TimelinePlanner,
)
from .minimax_h3_finite_segments import (
    MiniMaxH3FiniteLatentContinuation,
    MiniMaxH3FiniteAudioTrimTail,
    MiniMaxH3FiniteOutputTrim,
    MiniMaxH3FiniteSegmentFinalize,
    MiniMaxH3FiniteSegmentSampler,
    MiniMaxH3LockedAudioSlice,
    MiniMaxH3SilentAudioSlice,
    MiniMaxH3LockAudioLatent,
    MiniMaxH3LockedAudioMaster,
    MiniMaxH3SilentAudioMaster,
)
from .selflift_runtime import SelfLiftH3Sampler

# 节点类名保持原样（工作流存档靠它），只把菜单里的显示名换成中文
NODE_CLASS_MAPPINGS = {
    "WJZ_H3_TimelineDirector": MiniMaxH3TimelineDirector,
    "WJZ_H3_TimelinePlanner": MiniMaxH3TimelinePlanner,
    "WJZ_H3_TimelineEncoder": MiniMaxH3TimelineEncoder,
    "WJZ_H3_OmniPromptBridge": MiniMaxH3OmniPromptBridge,
    "WJZ_H3_FiniteSegmentSampler": MiniMaxH3FiniteSegmentSampler,
    "WJZ_H3_FiniteAudioTrimTail": MiniMaxH3FiniteAudioTrimTail,
    "WJZ_H3_FiniteOutputTrim": MiniMaxH3FiniteOutputTrim,
    "WJZ_H3_FiniteLatentContinuation": MiniMaxH3FiniteLatentContinuation,
    "WJZ_H3_FiniteSegmentFinalize": MiniMaxH3FiniteSegmentFinalize,
    "WJZ_H3_LockedAudioSlice": MiniMaxH3LockedAudioSlice,
    "WJZ_H3_SilentAudioSlice": MiniMaxH3SilentAudioSlice,
    "WJZ_H3_LockAudioLatent": MiniMaxH3LockAudioLatent,
    "WJZ_H3_LockedAudioMaster": MiniMaxH3LockedAudioMaster,
    "WJZ_H3_SilentAudioMaster": MiniMaxH3SilentAudioMaster,
    "WJZ_H3_TimelineSelfLiftSampler": SelfLiftH3Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WJZ_H3_TimelineDirector": "H3 导演台（一体化）",
    "WJZ_H3_TimelinePlanner": "H3 素材与分段计划台",
    "WJZ_H3_TimelineEncoder": "H3 分段编码",
    "WJZ_H3_OmniPromptBridge": "H3 全媒体提示词桥接",
    "WJZ_H3_FiniteSegmentSampler": "H3 分段采样（长视频）",
    "WJZ_H3_FiniteAudioTrimTail": "H3 尾部音频裁切（内部）",
    "WJZ_H3_FiniteOutputTrim": "H3 成片裁切（内部）",
    "WJZ_H3_FiniteLatentContinuation": "H3 分段接续（内部）",
    "WJZ_H3_FiniteSegmentFinalize": "H3 分段收尾（内部）",
    "WJZ_H3_LockedAudioSlice": "H3 锁定音轨切片（内部）",
    "WJZ_H3_SilentAudioSlice": "H3 静音切片（内部）",
    "WJZ_H3_LockAudioLatent": "H3 锁定音频 latent（内部）",
    "WJZ_H3_LockedAudioMaster": "H3 锁定音轨总轨（内部）",
    "WJZ_H3_SilentAudioMaster": "H3 静音总轨（内部）",
    "WJZ_H3_TimelineSelfLiftSampler": "H3 二采高清采样（内部）",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
