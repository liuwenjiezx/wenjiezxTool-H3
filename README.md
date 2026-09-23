# wenjiezxTool-H3

MiniMax H3 视频生成的导演台节点包，GPL-3.0 协议。

基于 [ComfyUI-MiniMaxH3-TimelineDirector](https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector) 整理，菜单显示名已中文化，节点类名未改动。

## 安装

把整个 `wenjiezxTool-H3` 文件夹放进 `ComfyUI\custom_nodes\`，重启 ComfyUI，搜索 `H3` 即可。

> 如果同时装着上游的 `ComfyUI-MiniMaxH3-TimelineDirector`，请先移除，避免同名节点冲突。

## 主要节点

| 类名 | 菜单显示名 |
|---|---|
| MiniMaxH3TimelinePlanner | H3 素材与分段计划台 |
| MiniMaxH3FiniteSegmentSampler | H3 分段采样（长视频） |
| MiniMaxH3TimelineEncoder | H3 分段编码 |
| MiniMaxH3TimelineDirector | H3 导演台（一体化） |
| MiniMaxH3OmniPromptBridge | H3 全媒体提示词桥接 |
| MiniMaxH3TimelineSelfLiftSampler | H3 二采高清采样 |

## 帧数规则

每段长度必须是 **5 + 17×n** 帧（5 / 22 / 39 / 56 …），最长 3592 帧。
段间过渡帧数为 **1** 或 **5 + 17×k**。
