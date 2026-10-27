# LostFleet Tools

可运行的 2D 角色骨骼动画素材拆分 MVP，使用 Python、Pillow、OpenCV、NumPy。浏览器中人工框选或绘制多边形，导出透明部件和坐标数据。无需云 API、账号或 GPU。

## 安装与运行

### Windows 独立运行（推荐，不依赖 Codex 预览）

准备 Windows 10/11、64 位 Python 3.12 和普通浏览器（Edge / Chrome / Firefox）。从 [Python 官方网站](https://www.python.org/downloads/windows/) 安装 Python，启用 Python Launcher 或勾选 **Add python.exe to PATH**。

**下载完整代码：** 在 [GitHub 工具分支](https://github.com/liusunyongtai/LostFleet-Tools/tree/codex/lostfleet-windows-mvp) 点击 **Code → Download ZIP**，然后完整解压。该工具位于 `codex/lostfleet-windows-mvp` 分支；下载默认 `main` 分支可能只有原始 README。不要直接双击压缩包内的脚本。解压后的项目根目录应同时包含 `start_windows.bat`、`requirements.txt`、`lostfleet_tools` 和 `tests`。

也可在安装 Git 后执行：

```bat
git clone --branch codex/lostfleet-windows-mvp https://github.com/liusunyongtai/LostFleet-Tools.git
cd LostFleet-Tools
```

双击 **`start_windows.bat`**。脚本会自动切换到自身目录，创建独立 `.venv`、安装固定版本依赖、检查依赖和导入，然后启动本机服务并打开默认浏览器。首次运行需联网下载依赖；运行时不需要云 API。脚本支持含空格和中文的目录，不需要管理员权限，也不需要手动激活虚拟环境或修改 PowerShell 执行策略。

在普通浏览器地址栏输入 **http://127.0.0.1:8000/**。这指向你自己的 Windows 电脑；云端的 8000 端口不会自动成为你本机的服务。不要直接打开 `index.html`，`file://` 无法提供图片与导出 API。保持命令窗口打开，按 Ctrl+C 停止服务。关闭前保存工程 JSON 和原图。

如果需要手动运行，在项目目录打开 CMD：

```bat
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -m lostfleet_tools.server --host 127.0.0.1 --port 8000 --open-browser
```

`requirements.txt` 仅包含 Pillow、OpenCV headless 和 NumPy 的固定版本。网页使用浏览器 Canvas，不需要 OpenCV GUI、Node.js 或 Playwright 才能使用。

**独立验证：**

```bat
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

启动后应看到完整三栏编辑器。上传 PNG/JPG，框选一个部件，点击“更新预览”，确认透明棋盘格，再导出 ZIP，检查 `full/`、`cropped/` 以及 `metadata.json`。点击“界面诊断”可核对构建 ID、`grid` 布局与 CSS 规则数。普通浏览器检查和 Windows 启动器的测试范围不同：云环境可以运行 Python/Chromium 测试，但没有 Windows 运行环境，不能声称已实机执行 `.bat`。

**常见问题：**

- 无法找到 Python：安装 64 位 Python 3.12，重新打开 CMD 后运行 `py -3.12 --version`。已有 `.venv` 版本不符时先保留或重命名，再让脚本重新创建。
- 依赖安装失败：查看脚本窗口的原始错误，确认可访问 PyPI；不要关闭 TLS 或签名验证。
- 端口占用或打开了其他应用：先关闭旧的 LostFleet 服务。也可手动使用 `--port 8001 --open-browser`，并打开相同端口的地址。
- 浏览器未自动打开：服务成功启动后手动输入上面的地址；不要关闭服务器窗口。
- 页面版本不符：关闭旧服务，从正确项目目录重新启动，用 Ctrl+F5 刷新，再查看“界面诊断”。服务响应的 `X-LostFleet-Build` 与页面构建 ID 应一致。
- 更新完整代码时先停止服务器，保留自己保存的工程和原图，再替换代码。代码包不含虚拟环境和用户素材。

### macOS / Linux

推荐 Python 3.12，在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m lostfleet_tools.server --port 8000
```

Windows 可用 `.venv\Scripts\activate` 激活。服务器启动后用本机浏览器访问 `http://127.0.0.1:8000`。服务默认仅监听本机；这是无认证、内存存储的单用户开发工具，不应直接暴露到公网。浏览器需要支持 Canvas、Pointer Events、Web Crypto 和下载功能。使用远程端口转发时请使用受信任的 HTTPS 地址，以保证工程 SHA-256 功能可用。

已配置的 Codex 云环境可直接运行，无需重新安装：

```bash
cd /workspace/LostFleet-Tools
source /workspace/.venvs/lostfleet-tools/bin/activate
python -m lostfleet_tools.server --port 8000
```

## 使用流程

1. 上传 PNG/JPG，最多 32 MiB、1600 万像素。JPEG 的 EXIF 旋转会先归正，后续坐标以归正后的图片为准。PNG 原始 Alpha 会被保留。
2. 从列表选择头部、身体、左右上臂、左右下臂、左右手、左右大腿、左右小腿、左右脚、武器或背包。左右以角色自身为准，不自动识别。
3. 矩形工具拖动框选；多边形工具逐点沿轮廓绘制，点击“完成多边形”或 Enter 闭合，Esc 取消。一个部件可添加多个区域。缩放到 200%/400% 可细化选区，坐标始终映射回原图。
4. 使用“擦除区域”绘制多边形或矩形，删除背景空洞或错误像素。“撤销”删除未完成多边形的最后一点，或最后一个已完成区域；“清空”重置当前部件和参数。修改已有区域可撤销重画，暂不支持拖动顶点。
5. 切换“枢轴点”工具并点击关节位置。不设置时默认使用添加区域外接框中心。边缘扩缩范围 −32～32 px，羽化 0～16 px；关节重叠半径 0～128 px，将枢轴附近的原图像素加入选区。调整按原图像素计算，每个部件独立保存。
6. 选择去白模式，点击“预览当前部件”检查棋盘格上的真实透明效果。参数改变会隐藏旧预览，需重新预览。默认删除与画布边缘连通的近白像素，保留被包围的白色衣物；内部背景空洞需手工擦除。“去除所有近白色”也会删除白色衣物，请谨慎使用。“保留原图 Alpha”适合已透明图片或完全手工绘制。阈值越低，去白范围越大。非白色复杂背景依靠多边形与擦除区域人工分离。
7. 点击“导出素材 ZIP”。仅导出有添加区域的部件；空选区、完全透明部件、非法坐标将明确报错，不会生成假成功文件。擦除优先级高于扩边、羽化和重叠。
8. “保存工程”导出包含选区、参数、枢轴与原图 SHA-256 的 JSON，便于继续编辑。载入前先上传相同原图。工程不包含原图，需自行保留。刷新页面会丢失未保存的选区；上传新图会清空旧选区，服务端仅保留最近上传的一张图片。

### 网页工作台

深色中文三栏布局：左侧管理部件，中央编辑原图，右侧调整属性、预览与导出。在 1920×1080 和 1366×768 下保持固定高度，部件列表和属性面板独立滚动，ZIP 导出按钮固定可见。

- 点击左侧部件行选择；眼睛开关显示/隐藏该部件的选区叠加，不改变导出范围。右侧“显示名称”可重命名界面标签，稳定英文 ID、PNG 文件名及 API 数据保持原样。
- 使用工具栏的矩形（R）、多边形（P）、擦除框（E）、枢轴（J）和平移（H）。精细擦除可用多边形配合“擦除选区”；枢轴支持点击和拖动。
- 滚轮以指针位置为中心缩放；也可使用底部比例选择、加减按钮。平移可使用平移工具、空格＋拖动或鼠标中键。“居中”同时重置平移和缩放，让图片适合画布宽高。
- 棋盘按钮切换透明背景显示。右侧更新 PNG 预览后，可在底部切换“分割预览”；“原图对比”切换未经蒙版处理的原始图片，对比期间只查看、不绘制。右侧预览自动放大至部件的可见范围，完整坐标和导出画布不变。
- 名称和显隐状态保存在工程 JSON 的可选 `ui` 字段，旧工程无需转换即可载入；原有 `schema_version`、部件结构、导出元数据格式与 Python API 均保留。
- Python 服务返回自包含 HTML：最新 CSS 和 JavaScript 内嵌在同一响应中，编辑器渲染不再依赖代理正确转发额外的样式/脚本请求。源码仍独立存放，原 `/style.css` 与 `/app.js` 端点保留正确 MIME。API 基址兼容带或不带末尾 `/` 的目录式预览路径。请通过 Python 服务访问工作台，而非直接打开 HTML 文件。
- HTML 的 `X-LostFleet-Build` 响应头及同名构建信息可用于核对代码版本；所有响应禁用缓存。“界面诊断”可查看当前窗口的路径（不含查询参数）、构建 ID、CSS 规则数、`getComputedStyle` 三栏布局、面板尺寸及检测到的 CSP 违规，并下载 JSON。若预览平台额外设置 CSP 或改写 HTML，仍需检查实际内嵌预览地址；本机检查不能替代该验证。

## 导出结构与坐标

```text
lostfleet-assets.zip
├── full/head.png          # 与原图等大，原始位置不变
├── cropped/head.png       # 按最终非零 Alpha 外接框裁切
├── ...                    # 其他已选部件
├── metadata.json
└── README.txt
```

所有 PNG 均为 RGBA，选区外 Alpha 为 0，完全透明像素 RGB 也归零。原图半透明像素会与选区、羽化蒙版相乘，不会被强制变为不透明。默认去白是阈值式处理，并非复杂背景抠图或边缘颜色去污染；原图已有压缩噪声、混色白边时需调整阈值、收边或擦除，再检查预览。

`metadata.json` 的 `schema_version` 为 1，包含画布大小、背景设置、坐标约定及部件列表：

| 字段 | 含义 |
| --- | --- |
| `name` | 稳定的英文部件名，如 `left_forearm` |
| `regions` | 原图坐标的添加/擦除多边形，可供后续处理复用 |
| `original_region_bounds` | 添加区域的几何外接框，不含边缘调整 |
| `canvas_origin` | 统一画布原点 `[0, 0]` |
| `crop_bounds` | 最终 Alpha 边界 `[left, top, right, bottom]`，右下界不包含 |
| `crop_offset` / `crop_size` | 裁切图在原画布中的左上位置 / 大小 |
| `pivot` / `pivot_in_crop` | 原图中的枢轴 / 裁切图中的枢轴，可能位于裁切图之外 |
| `pivot_spine` | 相对画布中心、Y 轴向上的枢轴坐标 |
| `adjustments` / `files` | 边缘、羽化、重叠设置 / 对应 PNG 路径 |

图像坐标为左上原点、X 向右、Y 向下，单位为像素。裁切版贴回 `crop_offset` 即可重建统一画布版。`pivot_spine = [pivot.x - width/2, height/2 - pivot.y]` 仅提供坐标换算参考。

ZIP 方便将图片作为附件导入 Spine，**不是 Spine 原生骨骼 JSON 或完整绑定工程**；需在 Spine 中自行创建骨骼、插槽、层级和附件绑定。当前不自动生成遮挡区域、不可见肢体、动画或权重。扩边与关节重叠仅复用原图已有像素，可能包含邻近部件，必须人工检查。

## 测试

在激活依赖环境后，从仓库根目录运行：

```bash
python -m unittest discover -s tests -v
```

覆盖 PNG/JPG 解码与 EXIF、透明 Alpha、去白模式、矩形/多边形与擦除、扩边/收边/羽化、关节重叠、坐标和裁切重建、ZIP、非法输入及真实 HTTP 上传→预览→导出。原有 `tests/test_image_pipeline.py` 也包含在内。

可选浏览器交互测试使用 Playwright 和 Chromium（不属于 Python 运行时依赖；当前云环境已提供），执行：

```bash
PYTHON=/workspace/.venvs/lostfleet-tools/bin/python node tests/browser_smoke.cjs
PYTHON=/workspace/.venvs/lostfleet-tools/bin/python node tests/ui_layout.cjs
# 检查已运行的 8000 端口，不另起 Python 服务：
LOSTFLEET_URL=http://127.0.0.1:8000/ PYTHON=/workspace/.venvs/lostfleet-tools/bin/python node tests/ui_layout.cjs
```

其他环境可将 `PYTHON` 指向当前 Python。浏览器优先使用 `CHROMIUM_PATH` 指定的路径，其次使用系统 `/usr/bin/chromium`，最后使用 Playwright 默认浏览器。该测试临时启动服务器，通过真实浏览器上传 JPG/PNG、缩放框选、多边形、擦除、枢轴与边缘调整、预览、工程保存载入和 ZIP 下载，并检查 ZIP 像素与坐标。截图保存在系统临时目录 `lostfleet-editor-smoke.png`，其余临时文件自动清理。

`ui_layout.cjs` 额外验证两种目标分辨率、自包含 HTML、CSS/JS 端点 MIME、页面无溢出、部件重命名/显隐、平移与指针定点缩放、拖动枢轴、原图对比、旧工程兼容，以及不带末尾斜杠的代理路径下上传和导出。截图保存为系统临时目录中的 `lostfleet-ui-1366.png`、`lostfleet-ui-1920.png`；指定运行中的端口后名称为 `lostfleet-live-8000-1366.png`、`lostfleet-live-8000-1920.png`，附同名 `-report.json` 诊断报告。使用程序生成的机器人测试图，不是实际游戏素材。代理回归使用测试代理，不能证明 Codex 实际内嵌预览成功；实际预览访问能力不可用时需明确报告。

## XQDZ 枪手 · Spine 动画 V2

本分支包含按授权提供的角色素材与五动作初稿，位于 [`examples/xqdz-spine`](examples/xqdz-spine/README.md)。下载 `codex/xqdz-spine-animations` 分支完整 ZIP 后，在本地 Spine 4.3.23 使用“导入数据”选择 `examples/xqdz-spine/xqdz-4.3.json`；图片目录是同级 `images`。先双击 `preview.html` 可查看 GIF。

包含 `idle`、`attack`（加强后坐、两倍火花与暖色补光）、`hit`、`death`、`move`。已验证官方 4.3 运行库加载与 167 个姿态，**未实机验证 Spine 4.3.23 桌面编辑器导入**。当前是整肢刚性绑定，没有膝肘弯曲、手部 IK 或遮挡补绘，详细限制见该目录说明。

## BSQ 毒囊怪物 · Spine 六动作与独立毒液

[`examples/bsq-spine`](examples/bsq-spine/README.md) 包含 `stand`、`dead`、`hit`、`attack_1`（向前抓击）、`attack_2`（喷吐）和原地循环 `move`，以及单独的 `bullet` 动画。Spine PRO 主版本使用连续加权网格处理原图合在一起的躯干与双腿，另附纯图片兼容版。目标编辑器为 4.3.23，官方运行库采样通过，但未实机测试桌面编辑器。独立动画包位于 `downloads/bsq-spine-4.3.23.zip`，无需启动网页工具即可导入本地 Spine。

## BTMG 四足机械犬 · Spine 五动作

[`examples/btmg-spine`](examples/btmg-spine/README.md) 包含 `stand`、`dead`、`hit`、`attack`（后撤蓄力后前冲撕咬）和原地四足循环 `move`。使用 22 根骨骼和连续加权网格处理四肢、牙区与尾部，目标版本为 Spine PRO 4.3.23。已做官方运行库的 383 个 60 Hz 姿态检查（含关键帧之间），未实机验证桌面编辑器导入。独立包为 `downloads/btmg-spine-4.3.23.zip`。

## 后续扩展接口

`lostfleet_tools/providers.py` 预留 `SegmentationProvider.suggest_regions()` 与 `InpaintingProvider.complete_occlusion()` 协议。未来模型建议应转换成可编辑的原图坐标选区；遮挡补绘应作为独立、可审阅的生成结果。MVP 未启用任何 AI 提供方，也不声称恢复遮挡部位。图像处理位于 `pipeline.py`，与本地 HTTP 界面解耦。


### GRYHY 感染宇航员 · 头部顶撞

新增 [Spine PRO 4.3.23 动画包说明](examples/gryhy-spine/README.md) 和 [五动作预览](examples/gryhy-spine/preview.html)：`stand`、`dead`、`hit`、`attack`、`move`。27 根骨骼、6 个加权网格，保留原始透明 PNG；移动原地循环，攻击以后仰蓄力、头盔前顶和回弹为主。可下载 [独立 ZIP](downloads/gryhy-spine-4.3.23.zip)。官方 4.3 运行库验证不等于桌面 Spine 实测，具体范围见包内说明。


### GRDGR 感染工人 · 前景手挥拳

新增 [Spine PRO 4.3.23 动画包说明](examples/grdgr-spine/README.md) 与 [五动作预览](examples/grdgr-spine/preview.html)：`stand`、`dead`、`hit`、`attack`、`move`。27 根骨骼、5 个加权网格，保留原始透明 PNG；移动原地循环，攻击包含收拳蓄力、向左挥拳、躯干跟进和回弹。下载 [独立 ZIP](downloads/grdgr-spine-4.3.23.zip)。官方 4.3 运行库检查范围及桌面 Spine 未实测的说明见包内 README。
