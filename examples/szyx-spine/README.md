# SZYX 四足异形 · Spine PRO 4.3.23

此包从本次上传的 `szyx.json` 与 16 张角色 PNG 制作，包含可编辑骨架、加权网格、六个动作、普通/手机尺寸 GIF、关键帧图、源文件校验值、构建脚本对应的配置与验证报告。原始 PNG 字节保持不变；没有自动生成或恢复遮挡的身体部分。

## 在本地 Spine 中打开

1. 完整解压 ZIP，保持 `szyx-4.3.json` 与 `images/` 相邻。
2. 在 **Spine PRO 4.3.23** 选择「导入数据」，导入根目录 **szyx-4.3.json**，比例 **1**。不要导入 `source/photoshop.json`，它是未绑定的 PS 源数据备份。
3. 图片路径应为 `./images/`；若图片丢失，在图片节点重新选择解压后的 `images` 文件夹。
4. 导入后另存为本地 `.spine` 工程。此包提供导入 JSON，不伪造专有 `.spine` 二进制工程。
5. 在动画模式选择下表动作；`stand` / `move` 开启循环，其余单次播放。导出游戏资源时，用本地编辑器输出与项目运行库匹配的 JSON/二进制和图集。不要用旧版本运行库加载 4.3 数据。

| 动作 | 时长 | 内容 |
|---|---:|---|
| stand | 2.4 秒 | 足端支撑、身体呼吸摇晃、六条触手错相蠕动、尾部延迟 |
| dead | 1.6 秒 | 失衡、屈腿趴落、回弹后静止；保持末帧 |
| hit | 0.6 秒 | 重心后移、头颚滞后、回弹恢复 |
| move | 1.2 秒 | 四足对角相位交替、抬脚迈步、身体起伏；原地循环 |
| attack | 1.2 秒 | 后收蓄力、张颚、身体前扑撕咬、闭颚和回弹 |
| attack_2 | 1.2 秒 | 单前爪向后高抬蓄力、向前抓击、三足支撑与回收 |

面向左侧。根节点在活体动作中不平移；攻击前伸由身体局部姿态产生，世界移动由游戏控制。死亡根节点有地面对齐修正。移动的支撑脚随传送带方向滑动，以便游戏驱动角色世界速度；在原地预览中这是正常的循环步态，不表示角色在场景中前进。

## 事件和使用建议

`attack` 在 0.53 秒发出 `bite_hit`，`attack_2` 在 0.50 秒发出 `claw_hit`；`hit` 在 0.10 秒发出 `impact`；`move` 在 0.60 / 1.20 秒发出 `footstep`；`dead` 在 1.40 秒发出 `death_end`。事件只是时机标记，不直接造成伤害。咬击参考 `bite_point`，抓击参考 `near_front_tip`；碰撞范围需要游戏设置。

建议活体动作切换混合约 0.10–0.16 秒，死亡约 0.08 秒；按游戏响应调整。攻击与受击末帧回到初始姿态，死亡保持倒地，不自动重置。GIF 为展示可能重复攻击，不代表游戏应循环播放攻击。

51 根骨骼，16 个独立加权网格。四条腿各含上段、下段和脚掌，足端解算烘焙成可编辑关键帧，不依赖运行时 IK 约束。头壳、牙齿、下颌为刚性绑定；尾部与触手多段延迟。只在现有拆层可覆盖范围内弯曲，没有补绘隐藏肢体。

## 查看预览与重新构建

`preview.html` 提供六组普通/手机 GIF；`previews/animation-contact.png` 为关键帧总览。可以在包目录执行 `python -m http.server 8000`，在普通浏览器打开 `http://127.0.0.1:8000/preview.html`。预览是独立软件渲染，不是 Spine 桌面截图。

完整仓库中执行（使用已有 Python 依赖）：

```bash
python tools/build_szyx_spine.py
SPINE_CORE_MODULE=/path/to/@esotericsoftware/spine-core/dist/index.js node tests/validate_szyx43.mjs
python -m unittest tests.test_szyx_spine -v
python -m unittest discover -s tests -v
```

配置为 `rig-config.json`；构建脚本位于仓库 `tools/build_szyx_spine.py`。官方运行库报告保存骨骼和采样顶点，可由 Python 独立几何计算复核。网格校验包含有序外轮廓、编辑器边索引、UV 三角形面积和连通性。

**验证范围：**本环境没有安装专有 Spine 桌面编辑器，**尚未实际执行 Spine PRO 4.3.23 桌面导入**。官方 4.3.13 JavaScript 运行库验证、拓扑检查和浏览器预览不能代替桌面导入验证。具体实际测试结果见 `verification.json`、`test-results.txt` 与 `runtime-validation.json`。若桌面导入报错，请提供新日志和报错附件名，以便定点排查。
