# SJT · attack_2 气流增强版 · Spine PRO 4.3.23

以此次上传的 `sjt-spine-4.3.23.rar` 中 **已由 Spine 导出的新版 JSON** 为基准，保留原有六个动作的曲线、原始骨骼和网格，仅为 `attack_2` 加入半透明气流与冲击气浪。没有用仓库中较早的 SJT 动作重新覆盖你的修改。


## 编辑器拓扑修复

根据导入日志中的 `Mesh has invalid triangles: airflow_15...19`，已将原来单个附件内的三条气流拆成三个独立插槽，每个网格只有一个 UV 区域，补齐 hull/edges，并统一 UV 与设置姿势中的三角形方向。原角色的骨骼、原始图集和六动作曲线/事件不变。旧版的单附件三重 UV 布局不能用运行库“顶点有限”检查来证明编辑器兼容。

更新后的数据已通过拓扑检查及官方运行库验证；**桌面 Spine 4.3.23 的实际导入仍待本机确认**。请使用本次修复包中的 JSON 和图集。

## 导入本地 Spine

1. 完整解压本包，保留 JSON、`images/` 和图集页。
2. 在 Spine PRO 4.3.23 选择“导入数据”，导入根目录 **`sjt-4.3.json`**，缩放填写 **1**，然后另存为新的 `.spine` 工程。
3. 在动画模式中选择 **`attack_2`** 播放。完整动作仍为原来的 1.4 秒。
4. `source/original-project/sjt.spine` 是你上传的原工程备份，字节未修改，**不含新增特效**。请导入根目录新版 JSON 查看效果；不能直接打开旧备份期待出现新特效。

当前环境不能启动桌面 Spine，因此没有修改或伪造二进制 `.spine`。新增内容已写入可编辑 JSON，并附配套图集。请在本机导入后保存自己的新版工程。

## 新增效果

- **0.50–约 0.667 秒：三条气流拖尾。** 用官方 Spine 4.3 运行库按 240 Hz 采样上传版本的拳端轨迹，生成沿路径弯曲的拖尾网格。下砸加速时出现，靠近最低点后渐隐。白色中心配少量冷蓝边缘，使用透明加法混合，不做火焰效果。
- **0.60–0.90 秒：冲击气浪。** 拳端接近动作最低点时，在地面附近形成短促、向外扩大的断环与气流线，随后消退。
- 特效放置有遮挡层次：弧形拖尾在人物后，地面气浪在前。效果不会把拳头盖成一团白色。
- `stand`、`move`、`hit`、`dead`、`attack_1` 的原曲线与事件不变，隐藏新增特效。`attack_2` 的原动作轨迹与 `slam_hit` 事件时间也没有改变。

新增 2 根辅助骨骼、4 个特效插槽、每条气流各 5 个拖尾帧网格（共 15 个）和 1 个冲击网格。共 19 根骨骼、12 个插槽。新增 PNG 是程序生成的透明特效，原始八张角色 PNG 不变，不补绘身体。

## 游戏运行时文件

根目录包含可一起使用的：

- `sjt-4.3.json`：含气流的六动作数据。
- `sjt-4.3.atlas.txt`：双页图集，保留原有八个区域，并追加两个特效区域。
- `sjt-4.3.png`：上传的原始角色图集页，字节不变。
- `sjt-airflow-fx.png`：新增特效图集页。

**必须同时带上两张图集 PNG。** 图集使用 PMA（预乘 Alpha），新增页也按 PMA 编码。请沿用与 Spine 4.3 匹配的运行库、PMA 与 additive 混合配置。编辑器导入则使用 `images/` 中的原图和直通 Alpha 特效图。

## 预览

打开 `preview.html` 查看：

- `previews/attack_2-comparison.gif`：左侧原上传动作，右侧新增特效，使用相同姿势。
- `previews/attack_2.gif`、`mobile-attack_2.gif`：正常和手机尺寸。
- `previews/airflow-contact.png`：发力、下砸和气浪消散关键帧。
- 其余动作的普通与手机尺寸预览。

GIF 的姿势来自官方 Spine 4.3 运行库，按其矩阵做独立软件渲染，**不是桌面 Spine 截图**。这样可以保留上传 JSON 的 Bezier 曲线与省略值，避免旧线性预览程序误读新版动作。攻击 GIF 为便于观看重复演示，游戏里仍按原设置播放。

可直接在普通浏览器打开 `preview.html`；若本机限制本地文件，运行 `python -m http.server 8000` 后打开 `http://127.0.0.1:8000/preview.html`。云端浏览器检查使用临时 HTTP 服务，不代表已测试本机 `file://` 或 Codex 内嵌预览。

## 验证与复现

原始上传数据、图集文本与工程备份位于 `source/`；角色贴图与原始文件 SHA256 清单也在其中。`source/original-runtime.json` 保存原动作运行库采样，`runtime-validation.json` 保存新数据采样。

在仓库根目录，准备好官方 `@esotericsoftware/spine-core 4.3.13` 后运行：

```sh
SPINE_CORE_MODULE=/absolute/path/to/@esotericsoftware/spine-core/dist/index.js SJT_AIRFLOW_BASE=1 SJT_AIRFLOW_REPORT=source/original-runtime.json node tests/validate_sjt_airflow43.mjs
python tools/build_sjt_airflow.py
SPINE_CORE_MODULE=/absolute/path/to/@esotericsoftware/spine-core/dist/index.js node tests/validate_sjt_airflow43.mjs
python tools/build_sjt_airflow.py --previews-only
python -m unittest discover -s tests -v
```

官方运行库直接加载实际双页 atlas，以 60 Hz 检查六动作 **448 个姿势**。原有 17 根骨骼的世界矩阵和原始网格采样点与上传版本逐帧相同。自动化另外检查曲线/事件原样保留、特效帧与拳端轨迹对齐、特效消退、移动端可见性、PMA 像素、图集区域和原图哈希。

**目标 Spine PRO 4.3.23；当前环境未实测桌面编辑器导入。** 官方同系列运行库已通过检查，仍请在本机导入并确认后保存工程。
