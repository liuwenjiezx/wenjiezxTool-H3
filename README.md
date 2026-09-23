# wenjiezxTool-H3

MiniMax H3 视频生成的**导演台节点包**，由 ComfyUI 服务器助手随环境一起安装。

> 许可：**GPL-3.0（开源）** —— 与本目录下的 `LICENSE-GPL.txt` 一致。

## 为什么这个包是开源的

H3 导演台的原始实现来自开源项目
[ComfyUI-MiniMaxH3-TimelineDirector](https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector)
（作者 Songssx，GPL-3.0）。

GPL-3.0 允许自由使用、修改、再分发，但要求衍生作品**同样以 GPL-3.0 开源**。
所以：

| 包 | 形态 | 许可 | 内容 |
|---|---|---|---|
| `wenjiezxTool` | 闭源 `.pyd` | 自有 + Apache-2.0 组件 | 自研工具节点、Krea2 编辑两件套、SAGE 注意力等 |
| **`wenjiezxTool-H3`**（本包） | **开源源码** | **GPL-3.0** | H3 导演台节点 |

两个包分开，各自合规 —— 把 GPL 的代码编译进闭源插件是违反许可的，分开就都没问题。

## 本包做了什么改动

- 整理目录：去掉上游的 `tests/`、示例与缓存文件，只留运行需要的部分；
- 菜单显示名中文化（**节点类名一个都没改**）；
- 补了这份中文说明。

节点逻辑未作功能性修改。

## 安装

把整个 `wenjiezxTool-H3` 文件夹放进 ComfyUI 的插件目录：

```
ComfyUI\custom_nodes\wenjiezxTool-H3\
    ├── __init__.py
    ├── minimax_h3_timeline_director.py
    ├── minimax_h3_finite_segments.py
    ├── drift_control_av.py
    ├── selflift_runtime\
    ├── js\                       # 导演台时间轴的前端（必须带上）
    ├── LICENSE-GPL.txt
    └── README.md
```

重启 ComfyUI 后在节点菜单里搜 `H3` 即可找到全部节点。

★**重要**：如果同时装着上游的 `ComfyUI-MiniMaxH3-TimelineDirector`，两边会注册
同名节点造成冲突。装本包前请先移除或改名上游那个目录。

## 节点一览

| 类名 | 菜单显示名 | 作用 |
|---|---|---|
| MiniMaxH3TimelinePlanner | H3 素材与分段计划台 | 解析时间轴、算对齐帧数、打包有序媒体 |
| MiniMaxH3FiniteSegmentSampler | H3 分段采样（长视频） | 把分段计划展开成采样图并逐段生成 |
| MiniMaxH3TimelineEncoder | H3 分段编码 | 编码单段的提示词与参考素材 |
| MiniMaxH3TimelineDirector | H3 导演台（一体化） | 一体化入口 |
| MiniMaxH3OmniPromptBridge | H3 全媒体提示词桥接 | 调用 Omni 提示词改写（需额外装改写器） |
| MiniMaxH3TimelineSelfLiftSampler | H3 二采高清采样 | 二采高清（内部） |
| 其余 `...Trim/Slice/Master` | 各类（内部） | 音频裁切、成片裁切、分段接续等 |

## 帧数规则（H3 硬性约束）

H3 的 latent 时间压缩是「17 帧一个单位 + 5 帧起始」，所以：

- 每段长度必须是 **5 + 17×n** 帧（5 / 22 / 39 / 56 …），最长 3592 帧（约 150 秒）
- 段间过渡帧数只认 **1** 或 **5 + 17×k**（实际可用 5 / 22）

不符合这个规则，计划台会直接报错而不是默默生成坏片。
