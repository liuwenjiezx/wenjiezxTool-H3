# wenjiezxTool-H3

MiniMax H3 视频生成的导演台节点包，GPL-3.0 协议。

基于 [ComfyUI-MiniMaxH3-TimelineDirector](https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector) 整理，菜单显示名已中文化，节点类名未改动。

> 2026-09-24：注册键已从 `MiniMaxH3*` 全部改为 `WJZ_H3_*`，与上游原版插件彻底错开——
> 两份同时安装也不会再触发"节点重复注册"冲突。旧工作流 JSON 里存的还是旧键名，
> 打开旧工作流若提示缺节点，重新选一次节点即可（机器上装着原版插件时会自动落到原版节点）。

## 安装

把整个 `wenjiezxTool-H3` 文件夹放进 `ComfyUI\custom_nodes\`，重启 ComfyUI，搜索 `H3` 即可。

> 如果同时装着上游的 `ComfyUI-MiniMaxH3-TimelineDirector`，无需移除——本包注册键已错开，两套节点可共存。

## 主要节点

| 注册键（工作流 JSON 里的 type） | 菜单显示名 |
|---|---|
| WJZ_H3_TimelinePlanner | H3 素材与分段计划台 |
| WJZ_H3_FiniteSegmentSampler | H3 分段采样（长视频） |
| WJZ_H3_TimelineEncoder | H3 分段编码 |
| WJZ_H3_TimelineDirector | H3 导演台（一体化） |
| WJZ_H3_OmniPromptBridge | H3 全媒体提示词桥接 |
| WJZ_H3_TimelineSelfLiftSampler | H3 二采高清采样 |

## 帧数规则

每段长度必须是 **5 + 17×n** 帧（5 / 22 / 39 / 56 …），最长 3592 帧。
段间过渡帧数为 **1** 或 **5 + 17×k**。

## 界面语言

2026-09-25 起，插件界面全量中文化，规则如下：

- **面板文案走私有词典**：`js/minimax_h3_timeline_director.js` 里维护 `TIMELINE_ZH`（中文，142 条）
  与 `TIMELINE_EN`（英文原版，保留备用），界面上所有字符串都通过 `tr("key")` 取值，
  改文案只改词典一处，不会漏。
- **自动识别语言**：读 ComfyUI 的 `Comfy.Locale` 设置，取不到时退回 `navigator.language`。
- **兜底顺序**：当前语言 → 中文 → 英文 → 键名本身。也就是说**即便漏翻，也会显示中文而不是英文**。
- **未知语言按中文处理**：只有已确认支持英文的语言（`en/ja/ko/fr/de/es/…`）才判定为英文，
  其余一律按中文显示——漏判成英文会比误判成中文麻烦得多。
- **Python 侧同步中文化**：节点的显示名、tooltip、description、报错文案、采样状态文字
  全部改为中文（含 `WJZ_H3_*` 系列 6 个节点与 SelfLift 内部节点）。
- 开发者控制台里的诊断日志（`Unable to load localization` 之类）保留英文，便于排查。
