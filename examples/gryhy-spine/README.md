# GRYHY · 感染宇航员骨骼动画

面向 **Spine PRO 4.3.23**，27 根骨骼、6 个加权网格，使用提供的六张透明 PNG。原始 PNG 字节不变；PS 的层顺序和初始位置保留，根骨骼移至脚下。没有补画被遮挡的身体，也没有新增鞋子；下装的裤脚作为落脚区域。

## 在本地 Spine 导入

1. 完整解压 ZIP，保留 `images/` 与 JSON 的相对目录。
2. 在 Spine PRO 4.3.23 中选择“导入数据”，选择 **gryhy-4.3.json**，缩放设为 **1**。不要导入 `source/photoshop.json`，那是备份原始摆位。
3. 若图片显示丢失，将骨骼图片路径指向解压目录的 `images/`。
4. 在动画模式选择下列动作。确认后用“另存为”保存为自己的 `.spine` 项目，再由 Spine 导出游戏运行时 JSON、atlas 和纹理。

本包是可编辑的 Spine JSON，并非原生 `.spine` 文件或已打包图集。

| 动作 | 时长 | 播放方式 | 设计 |
|---|---:|---|---|
| stand | 2.4 秒 | 循环 | 大头呼吸、头颈摆动、触须延迟跟随 |
| dead | 1.6 秒 | 单次，保持末帧 | 屈膝失衡、侧倒、落地回弹并稳定 |
| hit | 0.5 秒 | 单次 | 身体后震、头部延迟、回弹复位 |
| attack | 1.0 秒 | 单次 | 后仰压低蓄力，头盔朝画面左侧前顶，再回弹收势 |
| move | 1.0 秒 | 循环 | 双腿交替、躯干起伏、反向摆臂，根骨骼原地不漂移 |

强调大幅预备和跟随动作，兼顾小尺寸可读性；不是张嘴撕咬。四组感染触须在原头部 PNG 上通过网格加权跟随，没有切出新的图层。移动中的实际水平位移由游戏代码控制。

`attack` 在 **0.4 秒**发送 `headbutt_hit` 事件，`headbutt_point` 是头盔前顶的参考骨骼。游戏应在事件时使用自己的碰撞检测，不应只依赖骨骼坐标。`hit` 的 `impact` 在 0.1 秒，`dead` 的 `death_end` 在 1.4 秒；`move` 每半周期有一个 `footstep` 事件。

## 查看与修改

直接打开 `preview.html` 查看五动作 GIF、约 128px 角色高度的对比预览、关键姿势和骨骼图。预览是软件渲染，不是 Spine 桌面截图；手机预览为了比较会循环播放，实际死亡和受击应单次播放。

源码仓库的 `tools/build_gryhy_spine.py` 是生成器，`rig-config.json` 定义枢轴、网格绑定与肢体链。修改后运行：

```sh
python tools/build_gryhy_spine.py
SPINE_CORE_MODULE=/path/to/@esotericsoftware/spine-core/dist/index.js node tests/validate_gryhy43.mjs
python -m unittest discover -s tests -v
```

生成器复用仓库的通用网格渲染与 IK 工具。独立 ZIP 包可直接导入 Spine，不要求运行生成器。

## 验证范围

已使用官方 `@esotericsoftware/spine-core 4.3.13` 读取目标 4.3.23 JSON，以 60Hz 验证五动作共 **395 个姿势**的骨骼和全部网格顶点，并将抽样顶点与独立预览渲染器逐项比较。另检查原图哈希、透明度、PS 摆位、循环衔接、原地根骨骼、头部顶撞方向、裤腿网格连通、关键帧无整圈跳转、落地和动作复位。

**未运行桌面 Spine 4.3.23，不能保证桌面导入已经实测。** `runtime-validation.json` 明确记录 `desktopEditorTested: false`，请以本地导入结果为准。若需调整动画方向、枢轴或节奏，可继续在 PRO 的网格和骨骼轨道编辑。
