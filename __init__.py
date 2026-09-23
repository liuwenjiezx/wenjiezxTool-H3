# -*- coding: utf-8 -*-
"""wenjiezxTool-H3 —— MiniMax H3 导演台节点包（**开源 · GPL-3.0**）

这个包是做什么的
----------------
把「MiniMax H3 视频生成」用到的导演台节点收在一处，方便 ComfyUI 服务器助手
在给用户装环境时**一次性装上**，不用再让用户自己去 ComfyUI-Manager 里翻。

节点类名**一个都没改**（MiniMaxH3TimelinePlanner / MiniMaxH3FiniteSegmentSampler …），
所以现有的 H3 工作流 JSON 不用做任何修改，装完直接用。只把菜单里显示的名字
换成了中文。

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
    "MiniMaxH3TimelineDirector": MiniMaxH3TimelineDirector,
    "MiniMaxH3TimelinePlanner": MiniMaxH3TimelinePlanner,
    "MiniMaxH3TimelineEncoder": MiniMaxH3TimelineEncoder,
    "MiniMaxH3OmniPromptBridge": MiniMaxH3OmniPromptBridge,
    "MiniMaxH3FiniteSegmentSampler": MiniMaxH3FiniteSegmentSampler,
    "MiniMaxH3FiniteAudioTrimTail": MiniMaxH3FiniteAudioTrimTail,
    "MiniMaxH3FiniteOutputTrim": MiniMaxH3FiniteOutputTrim,
    "MiniMaxH3FiniteLatentContinuation": MiniMaxH3FiniteLatentContinuation,
    "MiniMaxH3FiniteSegmentFinalize": MiniMaxH3FiniteSegmentFinalize,
    "MiniMaxH3LockedAudioSlice": MiniMaxH3LockedAudioSlice,
    "MiniMaxH3SilentAudioSlice": MiniMaxH3SilentAudioSlice,
    "MiniMaxH3LockAudioLatent": MiniMaxH3LockAudioLatent,
    "MiniMaxH3LockedAudioMaster": MiniMaxH3LockedAudioMaster,
    "MiniMaxH3SilentAudioMaster": MiniMaxH3SilentAudioMaster,
    "MiniMaxH3TimelineSelfLiftSampler": SelfLiftH3Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3TimelineDirector": "H3 导演台（一体化）",
    "MiniMaxH3TimelinePlanner": "H3 素材与分段计划台",
    "MiniMaxH3TimelineEncoder": "H3 分段编码",
    "MiniMaxH3OmniPromptBridge": "H3 全媒体提示词桥接",
    "MiniMaxH3FiniteSegmentSampler": "H3 分段采样（长视频）",
    "MiniMaxH3FiniteAudioTrimTail": "H3 尾部音频裁切（内部）",
    "MiniMaxH3FiniteOutputTrim": "H3 成片裁切（内部）",
    "MiniMaxH3FiniteLatentContinuation": "H3 分段接续（内部）",
    "MiniMaxH3FiniteSegmentFinalize": "H3 分段收尾（内部）",
    "MiniMaxH3LockedAudioSlice": "H3 锁定音轨切片（内部）",
    "MiniMaxH3SilentAudioSlice": "H3 静音切片（内部）",
    "MiniMaxH3LockAudioLatent": "H3 锁定音频 latent（内部）",
    "MiniMaxH3LockedAudioMaster": "H3 锁定音轨总轨（内部）",
    "MiniMaxH3SilentAudioMaster": "H3 静音总轨（内部）",
    "MiniMaxH3TimelineSelfLiftSampler": "H3 二采高清采样（内部）",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
