# HTTP 识别服务说明

本文档说明当前仓库中的 `HTTP` 识别服务、测试脚本和代码结构。

相关文件：

- [server.py](/data/prj/yolov8_dial_reading/ultralytics/server.py)：兼容入口层，保留原有导出，供外部脚本和 ROS2 代码继续使用
- [recognition_http_server/app.py](/data/prj/yolov8_dial_reading/ultralytics/recognition_http_server/app.py)：HTTP 服务启动入口
- [recognition_http_server/service.py](/data/prj/yolov8_dial_reading/ultralytics/recognition_http_server/service.py)：核心任务编排层
- [test_http.py](/data/prj/yolov8_dial_reading/ultralytics/test_http.py)：本地联调脚本

## 快速开始

### http server

- conda环境部署：运行 `bash deploy_yolo_jetson_env.sh` 自动部署环境，同时会删除相关的安装包和 `.git` 节省空间，并添加自启的 `systemd` 服务；交付时可通过 `BUILD_PROTECTED=1` 启用核心源码编译
- 日志：`results/http_service/logs/server.log`
- 修改参数： `recognition_http_server/constants.py`

### ROS2 humble

- 功能暂不完全，暂无需求
- 先运行 ` bash ros2_recognition_service/deploy_ros2_ws.sh` 构建ros2工作区并编译
- 再运行 `source ./ros2_recognition_service/source_ros2_ws.bash`
- 接着ros2 launch 节点 `bash ros2_recognition_service/launch_recognition_service.sh`
- 可以在 `ros2_recognition_service/config/recognition.yaml` 中设置相关配置参数
- 详细介绍参考 `ros2_recognition_service/README.md `

## 1. 总体说明

当前服务支持 7 类识别任务：

- 表计读数
- 火源/烟雾检测
- 安全帽检测
- 消防设施检测
- 摔倒检测
- 灭火器检测
- 人车检测

整体流程：

1. 客户端向 `POST /api/v1/recognition/tasks` 提交任务。
2. 服务端立即返回 `processing` 状态。
3. 服务端后台线程逐张图执行识别。
4. 识别完成后，服务端向调用方回调 `POST /api/v1/recognition/callback`。
5. 客户端也可以通过 `GET /api/v1/recognition/tasks/{req_id}` 查询任务状态和回调结果。

## 2. 当前代码结构

为了便于后续继续增加检测类型，HTTP 服务已经拆成包结构：

```text
recognition_http_server/
  app.py
  config.py
  constants.py
  helpers.py
  http_api.py
  service.py
  schemas.py
  virtual_ptz.py
  dial_reading/
    __init__.py
    cli.py
    constants.py
    detections.py
    geometry.py
    pipeline.py
    visualization.py
    README.md
  handlers/
    detection.py
    meter.py
  utils/
    image_io.py
```

职责划分：

- `app.py`：启动入口
- `config.py`：命令行参数解析
- `constants.py`：默认参数和任务类型常量
- `helpers.py`：请求字段解析、任务类型路由、结果结构辅助函数
- `http_api.py`：HTTP 协议层
- `service.py`：任务状态、模型加载、任务调度
- `virtual_ptz.py`：无物理云台时使用静态图片模拟抓图，并记录 pan、tilt 和 zoom 请求
- `dial_reading/`：指针表计读数子包，承载原 `dial_reading.py` 中的模型推理、检测实例分配、几何读数和命令行能力
- `handlers/detection.py`：通用检测执行链路
- `handlers/meter.py`：表计相关执行链路
- `utils/image_io.py`：图片下载和本地路径准备

### 2.1 dial_reading 子包结构

`recognition_http_server/dial_reading/` 是指针表计读数逻辑的主实现目录。仓库根目录的 `dial_reading.py` 现在只保留兼容入口，继续支持旧脚本导入和 `python dial_reading.py` 命令。

```text
recognition_http_server/dial_reading/
  __init__.py
  cli.py
  constants.py
  detections.py
  geometry.py
  pipeline.py
  visualization.py
  README.md
```

职责划分：

- `__init__.py`：对外导出表计读数的公共入口，供 HTTP 服务和旧兼容层复用
- `cli.py`：命令行参数、单图运行、批量测试和终端输出逻辑
- `constants.py`：表计模型类别集合、9k 数据集重试尺寸等常量
- `detections.py`：YOLO 原始检测框整理、gauge 裁剪重试、点位框与表盘实例分配
- `geometry.py`：ROI 裁剪、中心估计、刻度点/指针尖端提取、圆弧比例计算
- `pipeline.py`：模型加载、推理编排、检测结果到读数实例的主流程
- `visualization.py`：结果图绘制和保存
- `README.md`：表计读数模块的细节说明

兼容性说明：

- `server.py` 当前仍然保留
- `server.py` 会重新导出 `RecognitionService`、`TaskState`、`make_error_payload`、`now_text` 和各类 `DEFAULT_*` 常量
- 这样做是为了不影响当前 ROS2 服务对 `server.py` 的依赖
- `dial_reading.py` 当前仍然保留
- `dial_reading.py` 会转发到 `recognition_http_server/dial_reading/` 子包，兼容旧的命令行用法和外部导入

## 3. 启动方式

推荐在仓库根目录下启动。

可用方式：

```bash
python server.py
```

也兼容直接运行：

```bash
python recognition_http_server/app.py
```

## 4. 启动参数

当前主要启动参数如下：

- `--host`：HTTP 监听地址，默认 `0.0.0.0`
- `--port`：HTTP 监听端口，默认 `3208`
- `--model`：表计模型路径
- `--fire-model`：火源检测模型路径
- `--safehat-model`：安全帽检测模型路径
- `--fire-protection-facilities-model`：消防设施检测模型路径
- `--person-fall-down-model`：摔倒检测模型路径
- `--fire-extinguisher-model`：灭火器检测模型路径
- `--person-and-cars-model`：人车检测模型路径
- `--imgsz`：推理尺寸
- `--conf`：置信度阈值
- `--device`：推理设备，例如 `cpu`、`0`
- `--min-value`：表计默认最小值
- `--max-value`：表计默认最大值
- `--result-root`：结果目录，默认 `results/http_service`
- `--callback-port`：回调端口，默认 `8088`
- `--request-timeout`：图片下载和回调超时时间
- `--ptz-align-enabled`：启用海康云台自动对齐，仅作用于指针表计任务
- `--ptz-controller`：云台后端，可选 `hikvision` 或 `virtual`，默认 `hikvision`
- `--virtual-ptz-image`：虚拟云台抓图时返回的静态图片路径，也可通过 `VIRTUAL_PTZ_IMAGE` 环境变量设置
- `--ptz-host` / `--ptz-port` / `--ptz-username` / `--ptz-password` / `--ptz-channel`：海康设备连接参数
- `--ptz-horizontal-fov-deg` / `--ptz-vertical-fov-deg`：手工视场角兜底值；服务端启用 PTZ 后会在每次指针表识别前优先通过海康 SDK 读取当前 FOV
- `--ptz-align-threshold-deg`：pan/tilt 偏移小于该阈值时不移动云台
- `--ptz-align-max-delta-deg`：单次 pan/tilt 修正的最大角度
- `--ptz-align-max-passes`：最多执行几轮“检测 gauge -> 调整云台 -> 重新抓图”
- `--ptz-nudge-*`：云台连续控制参数，用于把角度修正换算成 SDK start/stop 控制时长
- `--ptz-pan-nudge-degrees-per-second` / `--ptz-tilt-nudge-degrees-per-second`：当前云台和 speed 档位下实测的水平、垂直角速度；切换云台型号时优先调整这两个参数
- `--ptz-zoom-enabled` / `--no-ptz-zoom-enabled`：启用或关闭对齐后的光学变焦调整，默认启用
- `--ptz-zoom-*`：光学变焦目标比例、容差、轮次和控制时长参数

示例：

```bash
python server.py \
  --host 0.0.0.0 \
  --port 3208 \
  --model runs/weights/1_dial_reading/best.pt \
  --fire-model runs/weights/6_fire_and_smoke/best_fire.pt \
  --safehat-model runs/weights/7_safe_hat/best_person.pt \
  --device 0
```

## 5. 当前模型加载

服务启动时会一次性加载 7 个模型：

- 表计模型：默认 `runs/weights/1_dial_reading/best.pt`
- 火源/烟雾模型：默认 `runs/weights/6_fire_and_smoke/best_fire.pt`
- 安全帽模型：默认 `runs/weights/7_safe_hat/best_person.pt`
- 消防设施模型：默认 `runs/weights/8_fire_protection_facilities/fire-fighting-facilitie_best.pt`
- 摔倒模型：默认 `runs/weights/9_person_fall_down/fall_best.pt`
- 灭火器模型：默认 `runs/weights/10_fire_extinguisher/extinguisher_best.pt`
- 人车模型：默认 `runs/weights/11_person_and_cars/car_best.pt`

说明：

- `runs/weights/` 下目录名前缀对应外部请求的 `recognize_type`
- 表计任务当前使用 [recognition_http_server/dial_reading/](/data/prj/yolov8_dial_reading/ultralytics/recognition_http_server/dial_reading) 子包完成关键点检测、实例分配和几何后处理
- [dial_reading.py](/data/prj/yolov8_dial_reading/ultralytics/dial_reading.py) 只作为旧入口兼容层保留
- 火源、安全帽、消防设施、摔倒、灭火器和人车任务走通用 YOLO 检测链路

## 6. 任务类型路由规则

服务通过请求中的 `data_type[].recognize_type` 决定大类任务。

当前映射：

- `1`：表计
- `6`：火源/烟雾检测
- `7`：安全帽检测
- `8`：消防设施检测
- `9`：摔倒检测
- `10`：灭火器检测
- `11`：人车检测

兼容别名：

- `dict_meter_type` -> `1`
- `fire` -> `6`
- `safehat` -> `7`
- `person` -> `7`
- `fire_protection_facilities` -> `8`
- `person_fall_down` -> `9`
- `fire_extinguisher` -> `10`
- `person_and_cars` -> `11`

内部路由：

- `recognize_type` 先映射成内部 `task_kind`
- `task_kind` 再由 `RecognitionService` 中的 handler 注册表分发到具体处理函数

当前内置 handler：

- `meter`
- `fire`
- `safehat`
- `fire_protection_facilities`
- `person_fall_down`
- `fire_extinguisher`
- `person_and_cars`

这套结构的目的，是后续新增任务类型时不需要继续在主流程里堆 `if/elif`。

## 7. 表计子类型规则

表计任务除了 `recognize_type="1"` 外，还会进一步看 `recognize_subtype`。

当前规则：

1. 如果 `recognize_subtype` 命中保留的 OCR subtype 列表
   - 当前仅保留：`digital`
   - 走数码表链路

2. 如果 `recognize_subtype` 为空或为 `default`
   - 走指针表链路
   - 使用启动参数 `--max-value` 作为量程

3. 如果 `recognize_subtype` 不为空且不在 OCR subtype 列表中
   - 仍走指针表链路
   - 按 `float(recognize_subtype)` 解析量程

4. 如果既不是 OCR subtype，也无法转成浮点数
   - 服务端直接报参数错误

说明：

- 这意味着当前表计场景里，`recognize_subtype` 已经有了明确业务语义
- 数码表联调第一版约定传 `digital`
- 指针表联调时，后端可以直接传字符串化的量程，例如 `"3"`、`"5"`、`"25.0"`

## 8. 表计任务当前行为

### 8.1 指针表

指针表执行步骤：

1. 表计模型检测关键元素
2. 调用 `recognition_http_server/dial_reading/pipeline.py` 中的表计后处理流程
3. 得到归一化读数
4. 根据 `recognize_subtype` 解析得到量程
5. 返回 `归一化读数 * 量程` 作为最终读数

当前返回行为：

- `recognize_value`：最终读数，不再是单纯归一化值
- `confidence`：固定为 `"100"`
- `recognize_desc`：会带上归一化读数、量程、最终读数和 `arc_mode`

示例描述：

```text
识别成功，表计#1归一化读数=0.472000，量程=25.000000，最终读数=11.800000，arc_mode=...
```

### 8.2 指针表 PTZ 自动对齐与抓图

指针表计可以选择接入海康云台自动对齐。该能力只对表计任务中的指针表链路生效，不影响数码表、火源、安全帽、消防设施、摔倒、灭火器和人车检测任务。

启用方式：

```bash
python server.py \
  --ptz-align-enabled \
  --ptz-host 192.168.1.64 \
  --ptz-username admin \
  --ptz-password '<password>' \
  --ptz-channel 1
```

环境变量也可以提供部分海康连接参数：

- `HIK_HOST`
- `HIK_USERNAME`
- `HIK_PASSWORD`
- `HIK_LOCAL_IP`

完整流程：

1. 后端请求里可以在 `extra_info` 中携带当前云台状态，例如 `pan`、`tilt`、`zoom`。
2. 当前算法端不会使用这些状态值作为控制依据，只通过海康 SDK 读取和控制实际设备。
3. 服务先对输入图片做第一阶段 YOLO 推理，检测 `gauge`。
4. 服务通过海康 SDK 读取当前 FOV，选取置信度最高的 `gauge`，根据表盘中心相对画面中心的偏移计算 pan/tilt 修正量。
5. 如果偏移超过 `--ptz-align-threshold-deg`，通过 `HikvisionPTZController` 调用海康 SDK 控制云台。
6. 云台稳定后重新抓图，保存为 `*_align_1.jpg`、`*_align_2.jpg` 等，再重新检测 `gauge`。
7. pan/tilt 对齐结束后，如果启用 zoom，会根据表盘高度占画面高度的比例决定 zoom in 或 zoom out。
8. zoom 后重新抓图，保存为 `*_zoom_1.jpg`、`*_zoom_2.jpg` 等，再重新检测 `gauge`。
9. 最终对齐后的抓拍图保存为 `*_aligned`，后续指针读数基于这张最终图继续执行。
10. 最终图中的 `gauge` 会被裁剪并 resize 到 640x640，再进行第二阶段关键点检测和几何读数。

中间抓图保存位置：

```text
results/http_service/outputs/<req_id>/<stem>_align_1.jpg
results/http_service/outputs/<req_id>/<stem>_align_2.jpg
results/http_service/outputs/<req_id>/<stem>_zoom_1.jpg
```

最终对齐抓拍图保存规则：

- 本地输入：写到请求 `image_path` 同目录，在扩展名前追加 `_aligned`
- HTTP/HTTPS URL 输入：无法写回远端，写到 `results/http_service/outputs/<req_id>/`

示例：

```text
/data/test/meter.jpg
/data/test/meter_aligned.jpg
```

HTTP 服务返回规则：

- 指针表识别成功且 `_aligned` 文件存在时，`data_result[].image_path` 返回最终对齐抓拍图路径
- `data_result[].image_path_result` 返回最终结果图路径
- 本地输入成功时，结果图仍会额外复制为原图同目录的 `*-detection.jpg`
- 识别失败时，`image_path` 和 `image_path_result` 都保持原请求路径

相关实现入口：

- `recognition_http_server/service.py`：创建 `PTZAlignmentConfig` 和 `HikvisionPTZController`，并为 meter handler 注入抓图保存路径
- `recognition_http_server/hikvision_ptz.py`：海康 SDK 登录、读取 PTZ/FOV、pan/tilt/zoom 控制和 HTTP 抓图
- `recognition_http_server/ptz_alignment.py`：像素偏移到 pan/tilt 角度、表盘高度到 zoom 请求的计算
- `recognition_http_server/dial_reading/pipeline.py`：在 `meter_data_9k` 指针表流程中执行对齐、变焦、抓图保存和最终读数
- `scripts/test_hikvision_ptz_alignment.py`：独立现场验证脚本，可用于不启动 HTTP 服务时验证云台对齐效果

维护注意：

- 后续调整云台控制优先改 `recognition_http_server/hikvision_ptz.py` 和 `recognition_http_server/ptz_alignment.py`
- 后续调整表计读数优先改 `recognition_http_server/dial_reading/`
- 不要在 `service.py` 里继续增加新的任务 `if/elif` 分支，新增任务应走 handler 注册表
- PTZ 对齐依赖 `meter_data_9k` 模型先检测到 `gauge`；找不到 `gauge` 会导致该图表计任务失败
- 单张图内某个表盘几何读数失败会下沉到该表盘实例，不会主动让其他表盘失败

### 8.3 海康云台现场部署与标定 SOP

新设备首次部署时，先运行部署脚本，再根据实际云台型号微调 pan、tilt 和 zoom 参数。实时 FOV 由 SDK 自动读取，不需要手工标定。

#### 8.3.1 准备部署资源

进入项目目录并确认环境包、表计模型和 ARM64 SDK 存在：

```bash
cd /home/glr/glr_nav_perception/yolo_detection
ls resources/yolo-jetson.tar.gz
ls runs/weights/1_dial_reading/best.pt
ls HK_SDK/HK_SDK_arm64_Linux/lib/linux/libhcnetsdk.so
```

#### 8.3.2 执行部署脚本

```bash
bash deploy_yolo_jetson_env.sh
```

部署脚本会自动检测 CPU 架构，选择 ARM64 或 x86 海康 SDK，部署 conda 环境，交互式读取云台连接参数，并安装 `ai_detection.service`。交付时可通过 `BUILD_PROTECTED=1` 额外编译核心源码。云台连接参数保存到：

```text
recognition_http_server/hikvision.env
```

部署完成后检查服务：

```bash
systemctl status ai_detection.service --no-pager
curl -fsS http://127.0.0.1:3208/health
```

#### 8.3.3 验证云台连接和实时 FOV

```bash
set -a
source recognition_http_server/hikvision.env
set +a

conda run -n yolo-jetson python scripts/hk_sdk_probe.py \
  --host "$HIK_HOST" \
  --port "$HIK_PORT" \
  --username "$HIK_USERNAME" \
  --password "$HIK_PASSWORD" \
  --channel "$HIK_CHANNEL" \
  --read-gis-fov
```

预期输出包含：

```text
login_ok ...
ptz_ok ... pan=... tilt=... zoom=...
gis_fov_ok ... hfov=... vfov=... zoom=...
```

登录失败时，依次检查设备 IP、SDK 端口、账号密码，以及是否需要配置 `HIK_LOCAL_IP` 绑定指定网卡。

#### 8.3.4 标定 pan 和 tilt 角速度

不同型号云台在相同 SDK speed 档位下的实际转速不同。建议固定使用 `--nudge-speed 3`，分别测试右、左、上、下四个方向，每次移动 `0.5s`。

以向右测试为例：

```bash
conda run -n yolo-jetson python scripts/hk_sdk_probe.py \
  --host "$HIK_HOST" \
  --port "$HIK_PORT" \
  --username "$HIK_USERNAME" \
  --password "$HIK_PASSWORD" \
  --channel "$HIK_CHANNEL" \
  --nudge-direction right \
  --nudge-duration 0.5 \
  --nudge-speed 3
```

每次运行都会输出移动前后的 PTZ 位置。按以下公式计算角速度：

```text
方向角速度 = abs(移动后角度 - 移动前角度) / 0.5
pan_dps = 左右方向角速度平均值
tilt_dps = 上下方向角速度平均值
```

当前现场设备的参考值：

```text
pan_dps = 12.0
tilt_dps = 5.5
```

#### 8.3.5 持久化设备参数

推荐通过 systemd override 保存现场参数，避免重新部署代码后丢失标定值：

```bash
sudo systemctl edit ai_detection.service
```

填写：

```ini
[Service]
ExecStart=
ExecStart=/bin/bash -lc 'source "/home/glr/miniconda3/etc/profile.d/conda.sh" && conda activate "/home/glr/miniconda3/envs/yolo-jetson" && exec python "/home/glr/glr_nav_perception/yolo_detection/server.py" --ptz-nudge-speed 3 --ptz-zoom-max-ratio 4 --ptz-pan-nudge-degrees-per-second 12.0 --ptz-tilt-nudge-degrees-per-second 5.5 --ptz-align-max-delta-deg 2.0'
```

应用配置：

```bash
sudo systemctl daemon-reload
sudo systemctl restart ai_detection.service
curl -fsS http://127.0.0.1:3208/health
```

#### 8.3.6 提交表计任务并微调

提交代表性的指针表任务，同时观察日志：

```bash
tail -f results/http_service/logs/server.log
```

重点关注：

```text
meter ptz fov/tuning read from sdk:
hikvision ptz aligned:
meter ptz alignment checked:
```

理想状态是表盘在 1 到 3 轮内逐渐接近画面中心，不在目标两侧反复跳动。参数调整规则：

| 现象              | 调整方式                                                    |
| ----------------- | ----------------------------------------------------------- |
| pan 每次移动过量  | 增大 `--ptz-pan-nudge-degrees-per-second`                   |
| pan 移动不足      | 减小 `--ptz-pan-nudge-degrees-per-second`                   |
| tilt 每次移动过量 | 增大 `--ptz-tilt-nudge-degrees-per-second`                  |
| tilt 移动不足     | 减小 `--ptz-tilt-nudge-degrees-per-second`                  |
| 初次跳动幅度太大  | 减小 `--ptz-align-max-delta-deg`                            |
| 中心附近频繁微调  | 适当增大 `--ptz-align-threshold-deg`                        |
| zoom 超过设备能力 | 设置 `--ptz-zoom-max-ratio`                                 |
| zoom 过快         | 减小 `--ptz-zoom-nudge-seconds` 或 `--ptz-zoom-nudge-steps` |

每轮参数调整建议控制在 `10%` 到 `20%`。

#### 8.3.7 验收清单

```bash
systemctl is-active ai_detection.service
curl -fsS http://127.0.0.1:3208/health
```

逐项确认：

- SDK 登录成功，能够读取实时 FOV
- 当前 zoom 在设备支持范围内
- pan、tilt 不再明显移动过量
- 连续提交多次表计任务时，云台能够稳定收敛
- 重启设备后服务自动启动，现场标定参数仍然生效

### 8.4 数码表

数码表链路目前只完成了服务端骨架，尚未真正接入 ROI 检测和 OCR。

当前行为：

- 当 `recognize_subtype="digital"` 时，会进入数码表处理入口
- 当前会返回一条失败结果，提示 OCR 链路尚未接入
- 同时保存原图到结果图路径，方便后续联调

这部分是为后续接入 `ROI 检测 -> OCR` 链路预留的扩展口

## 9. 纯检测任务当前行为

火源/烟雾、安全帽、消防设施、摔倒、灭火器和人车检测都复用通用检测执行器。

执行步骤：

1. 使用对应 YOLO 模型对整图推理
2. 将检测框绘制到结果图
3. 每个检测框输出一条 `recognize_data`

返回规则：

- `recognize_value`：类别名
- `confidence`：置信度百分比整数
- `recognize_image_index`：第几个检测框

如果没有检测到目标：

- 返回一条失败记录
- `recognize_image_index = "0"`
- `recognize_value = ""`
- `confidence = "0"`

## 10. 请求接口

### 10.1 提交任务

- 方法：`POST`
- 路径：`/api/v1/recognition/tasks`

请求体示例：

```json
{
  "req_id": "demo-0001",
  "image_path": "/data/test/a.jpg,/data/test/b.jpg",
  "data_type": [
    {
      "recognize_type": "1",
      "recognize_subtype": "3"
    },
    {
      "recognize_type": "6",
      "recognize_subtype": ""
    }
  ],
  "extra_info": "{\"debug_center\": false}"
}
```

字段说明：

- `req_id`：任务 ID，可传可不传
- `image_path`：图片路径，支持本地路径或 HTTP URL，多张图用英文逗号分隔
- `data_type`：识别项列表
- `extra_info`：附加参数，支持对象或 JSON 字符串

### 10.2 查询任务

- 方法：`GET`
- 路径：`/api/v1/recognition/tasks/{req_id}`

### 10.3 健康检查

- 方法：`GET`
- 路径：`/health`

## 11. data_type 处理规则

`data_type` 是一个数组，每一项描述一张图对应的一种识别需求。

规则如下：

- 如果未传 `data_type`
  - 默认等价于 `[{"recognize_type": "1", "recognize_subtype": "default"}]`

- 如果 `data_type` 只有 1 项
  - 会复用到所有图片

- 如果 `data_type` 有多项
  - 数量必须与图片数量一致
  - 按顺序一一对应

当前约束：

- 一张图只做一种识别
- 当前不支持“同一张图叠加多个识别类型”

## 12. 图片输入与输出

图片输入支持两类：

- 本地路径
- HTTP/HTTPS URL

处理方式：

- 如果是 URL，会先下载到 `results/http_service/inputs/<req_id>/`
- 如果是本地文件，会直接读取

结果输出目录：

```text
results/http_service/outputs/<req_id>/
```

额外结果保存规则：

- 如果请求中的 `image_path` 是本地路径，服务会额外复制一份结果图到原图同目录
- 文件名规则为在扩展名前追加 `-detection`
- 例如 `/data/test/a.jpg` 会额外生成 `/data/test/a-detection.jpg`
- 如果请求中的 `image_path` 是 URL，则无法写回远端路径，仍只保留 `results/http_service/outputs/<req_id>/` 下的结果图

结果图命名规则：

- 表计：`*_result_meter.jpg`
- 火源：`*_result_fire.jpg`
- 安全帽：`*_result_safehat.jpg`
- 消防设施：`*_result_fire_protection_facilities.jpg`
- 摔倒：`*_result_person_fall_down.jpg`
- 灭火器：`*_result_fire_extinguisher.jpg`
- 人车：`*_result_person_and_cars.jpg`

指针表 PTZ 对齐额外文件：

- 中间抓图：`*_align_1.jpg`、`*_align_2.jpg`、`*_zoom_1.jpg`
- 最终对齐抓拍图：本地输入为原图同目录 `*_aligned.<ext>`；URL 输入为 `results/http_service/outputs/<req_id>/*_aligned.jpg`

## 13. 返回结构

回调返回顶层结构示例：

```json
{
  "req_id": "demo-0001",
  "data_result": [
    {
      "image_path": "/data/test/a.jpg",
      "image_path_result": "/data/test/a-detection.jpg",
      "recognize_data": [
        {
          "recognize_image_index": "1",
          "recognize_type": "6",
          "recognize_subtype": "",
          "recognize_value": "fire",
          "confidence": "92",
          "recognize_desc": "识别成功，火源检测检测到fire，置信度=0.9231"
        }
      ]
    }
  ],
  "code": 0,
  "resp_msg": "Task finished successfully.",
  "finish_time": "2026-04-14 10:00:00"
}
```

`recognize_data` 每项包含：

- `recognize_image_index`
- `recognize_type`
- `recognize_subtype`
- `recognize_value`
- `confidence`
- `recognize_desc`

字段语义：

- `image_path`
  - 普通任务：原请求图片路径
  - 指针表 PTZ 对齐成功：最终 `_aligned` 抓拍图路径

- `recognize_image_index`
  - 表计：第几个表计实例
  - 检测：第几个检测框
  - 未检出：固定 `"0"`

- `recognize_type`
  - 当前请求的识别类型

- `recognize_subtype`
  - 当前请求的子类型或量程参数

- `recognize_value`
  - 指针表：最终读数
  - 数码表：后续会是 OCR 结果
  - 检测：类别名

- `confidence`
  - 指针表：当前固定 `"100"`
  - 检测：置信度百分比整数

- `recognize_desc`
  - 面向业务的说明文字

## 14. 任务状态查询返回

查询接口返回中包含：

- `task_status`
- `created_at`
- `updated_at`
- `callback_payload`
- `callback_status_code`
- `callback_error`
- `error`

可用于：

- 查看任务是否完成
- 查看服务端实际回调内容
- 排查回调失败或识别异常

## 15. test_http.py 联调脚本

`test_http.py` 用于本地端到端联调，职责包括：

- 启动本地回调 HTTP 服务
- 向识别服务提交任务
- 等待回调
- 校验回调结构
- 将回调结果落盘
- 在终端打印结果摘要

### 15.1 启动方式

示例：

```bash
python test_http.py \
  --server http://127.0.0.1:3208 \
  --images /data/test/a.jpg \
  --data-type 6
```

也支持把类型和 subtype 拆开传：

```bash
python test_http.py \
  --server http://127.0.0.1:3208 \
  --images /data/test/a.jpg \
  --recognize-type 1 \
  --recognize-subtype 25
```

如果现场想直接从海康摄像机抓一张图再提交识别，可以使用：

```bash
python test_http.py \
  --server http://127.0.0.1:3208 \
  --hik-capture \
  --hik-host 192.168.1.64 \
  --hik-password '<password>' \
  --data-type 1:25
```

### 15.2 常用参数

- `--server`：识别服务地址，默认 `http://127.0.0.1:3208`
- `--images`：一个或多个图片路径
- `--data-type`：格式为 `type[:subtype]`
- `--recognize-type`：显式传识别类型，可重复传入
- `--recognize-subtype`：显式传 subtype，可传一个共享值，也可按类型数量逐项传入
- `--scene`：测试预设场景
- `--callback-host`
- `--callback-port`
- `--callback-path`
- `--debug-center`
- `--timeout`
- `--output`
- `--hik-capture`：先从海康摄像机抓图，再把抓到的本地图片作为 `image_path` 提交
- `--hik-host` / `--hik-port` / `--hik-username` / `--hik-password` / `--hik-channel`：海康抓图连接参数
- `--hik-output-dir`：抓图保存目录，默认 `results/http_service/hikvision_captures`
- `--hik-snapshot-name`：抓图文件名；不传时自动生成 `hik_YYYYmmdd_HHMMSS.jpg`
- `--hik-restore-delay`：测试结束后等待多少秒再回退到初始 PTZ 位置，默认 `10`
- `--no-hik-restore`：关闭测试结束后的 PTZ 自动回退

说明：

- 不使用 `--hik-capture` 时，`--images` 仍然必填
- 使用 `--hik-capture` 时，脚本会忽略手工传入的 `--images`，只提交当前抓到的那一张图
- 使用 `--hik-capture` 时，脚本会在提交任务前记录初始 pan/tilt/zoom，回调处理完成后默认等待 10 秒并恢复到初始位置
- `--hik-host`、`--hik-username`、`--hik-password` 可分别通过 `HIK_HOST`、`HIK_USERNAME`、`HIK_PASSWORD` 环境变量提供

### 15.3 参数传法说明

`test_http.py` 当前支持两种传法：

第一种：紧凑写法，使用 `--data-type type[:subtype]`

- `--data-type 1:3`
- `--data-type 1:25`
- `--data-type 1:digital`
- `--data-type 6`
- `--data-type 7`
- `--data-type 8`
- `--data-type 9`
- `--data-type 10`
- `--data-type 11`
- `--data-type meter:default`
- `--data-type fire`
- `--data-type safehat`
- `--data-type fire_protection_facilities`
- `--data-type person_fall_down`
- `--data-type fire_extinguisher`
- `--data-type person_and_cars`

第二种：拆分写法，显式使用 `--recognize-type` 和 `--recognize-subtype`

- `--recognize-type 1 --recognize-subtype 25`
- `--recognize-type 1 --recognize-subtype digital`
- `--recognize-type 6`
- `--recognize-type 8`
- `--recognize-type 9`
- `--recognize-type 10`
- `--recognize-type 11`
- `--recognize-type fire`
- `--recognize-type fire_protection_facilities`
- `--recognize-type person_fall_down`
- `--recognize-type fire_extinguisher`
- `--recognize-type person_and_cars`
- `--recognize-type 1 --recognize-type 6 --recognize-subtype 25 --recognize-subtype ""`

规则说明：

- `--data-type` 和 `--recognize-type/--recognize-subtype` 不能混用
- `--recognize-subtype` 可以不传
- `--recognize-subtype` 只传 1 次时，会复用到所有 `--recognize-type`
- `--recognize-subtype` 也可以按 `--recognize-type` 数量一一对应传入

- 表计不写 subtype 时，测试脚本会自动补成 `default`
- 如果测数码表，建议显式写成 `1:digital`

### 15.4 scene 预设

如果没有显式传 `--data-type`，会按 `--scene` 自动填默认值。

当前预设：

- `meter`
- `fire`
- `safehat`

说明：`test_http.py` 的 `--scene` 预设目前只覆盖以上三个常用场景；新增检测类型可以显式传 `--data-type 8`、`--data-type 9`、`--data-type 10` 或 `--data-type 11`。

注意：

- 当前 `meter` 预设里仍然使用了示例量程值
- 如果你要验证当前指针表逻辑，建议显式传 `--data-type 1:<量程>`，或 `--recognize-type 1 --recognize-subtype <量程>`

## 16. 当前已知限制

- 数码表 OCR 还未真正接入，只保留了服务骨架
- 当前一张图只支持一种识别任务
- 任务回调地址仍然由客户端 IP 和固定回调端口拼接得到
- 导入 `dial_reading.py` 时可能会看到 `matplotlib` 缓存目录 warning，但不影响服务运行

## 17. 推荐联调方式

1. 在仓库根目录启动服务：

```bash
python server.py
```

2. 使用 `test_http.py` 提交任务：

火源检测：

```bash
python test_http.py \
  --server http://127.0.0.1:3208 \
  --images /data/test/fire.jpg \
  --data-type 6
```

安全帽检测：

```bash
python test_http.py \
  --server http://127.0.0.1:3208 \
  --images /data/test/safehat.jpg \
  --data-type 7
```

消防设施检测：

```bash
python test_http.py \
  --server http://127.0.0.1:3208 \
  --images /data/test/fire_protection_facilities.jpg \
  --data-type 8
```

摔倒检测：

```bash
python test_http.py \
  --server http://127.0.0.1:3208 \
  --images /data/test/fall.jpg \
  --data-type 9
```

灭火器检测：

```bash
python test_http.py \
  --server http://127.0.0.1:3208 \
  --images /data/test/extinguisher.jpg \
  --data-type 10
```

人车检测：

```bash
python test_http.py \
  --server http://127.0.0.1:3208 \
  --images /data/test/person_and_cars.jpg \
  --data-type 11
```

指针表读数：

```bash
python test_http.py \
  --server http://127.0.0.1:3208 \
  --images /data/test/meter.jpg \
  --data-type 1:25
```

或：

```bash
python test_http.py \
  --server http://127.0.0.1:3208 \
  --images /data/test/meter.jpg \
  --recognize-type 1 \
  --recognize-subtype 25
```

数码表骨架联调：

```bash
python test_http.py \
  --server http://127.0.0.1:3208 \
  --images /data/test/digital.jpg \
  --data-type 1:digital
```

或：

```bash
python test_http.py \
  --server http://127.0.0.1:3208 \
  --images /data/test/digital.jpg \
  --recognize-type 1 \
  --recognize-subtype digital
```

## 18. 核心源码编译与交付

交付到 Jetson Orin NX 时，可以使用 Cython 将核心业务模块编译为 Python 扩展模块 `.so`，提高源码反编译成本。入口、配置和兼容层继续保留为 Python 文件，便于现场维护。

注意：

- `.so` 与 CPU 架构、Python ABI 和系统环境绑定
- ARM64 Jetson 上生成的 `.so` 不能直接用于 x86，也不能跨 Python 小版本复用
- 编译不能提供绝对不可逆保护，模型权重仍需按项目交付要求单独管理

### 18.1 编译环境

必须在目标 Jetson 或相同 JetPack、ARM64 架构和 Python 版本的构建机上执行。项目部署环境默认使用 `yolo-jetson`：

```bash
cd /home/glr/prj/yolo_detection
gcc --version
/home/glr/miniconda3/envs/yolo-jetson/bin/python -m pip install Cython
/home/glr/miniconda3/envs/yolo-jetson/bin/python scripts/build_protected.py build_ext --inplace
```

系统必须预先安装 GCC。缺失时安装：

```bash
sudo apt install build-essential
```

构建脚本：

```text
scripts/build_protected.py
```

脚本会编译服务编排、请求辅助、海康和虚拟云台、handler、表计后处理以及图片输入模块。Cython 中间文件统一写入：

```text
build/cython/
```

生成的扩展模块类似：

```text
recognition_http_server/service.cpython-310-aarch64-linux-gnu.so
recognition_http_server/virtual_ptz.cpython-310-aarch64-linux-gnu.so
recognition_http_server/dial_reading/pipeline.cpython-310-aarch64-linux-gnu.so
```

Python 会优先加载同名 `.so`。仅手工编译时，上述命令默认保留 `.py`，方便继续调试。准备交付包时增加 `--remove-sources`：

```bash
/home/glr/miniconda3/envs/yolo-jetson/bin/python \
  scripts/build_protected.py build_ext --inplace --remove-sources
```

该参数会先确认每个受保护模块都已生成对应 `.so`，再删除核心模块对应的 `.py`、字节码缓存和 `build/` 下的 Cython 中间文件。入口文件、配置文件和各包的 `__init__.py` 会保留。

`deploy_yolo_jetson_env.sh` 默认跳过编译，方便开发阶段直接调试 Python 源码。交付或专项验证时，显式启用保护构建：

```bash
BUILD_PROTECTED=1 bash deploy_yolo_jetson_env.sh
```

启用后，脚本会先检查系统 GCC，再检查 `yolo-jetson` 环境中的 Cython；Cython 缺失时自动尝试通过 pip 安装。随后编译 `.so`，删除受保护的原始源码和中间文件，验证关键模块确实从 `.so` 加载，再继续重启 systemd 服务。

注意：`BUILD_PROTECTED=1` 面向最终交付，会删除核心源码。开发板仍需继续修改和调试代码时，不要开启该选项。

开发调试时直接运行即可，或显式保持纯 Python 模式：

```bash
BUILD_PROTECTED=0 bash deploy_yolo_jetson_env.sh
```

### 18.2 无物理云台测试

测试板没有物理云台时，使用虚拟云台后端：

```bash
python server.py \
  --ptz-controller virtual \
  --virtual-ptz-image /home/glr/prj/yolo_detection/data/105.jpg
```

如果服务通过 systemd 启动，且 `ai_detection.service` 引用了 `recognition_http_server/hikvision.env`，该文件必须存在。无物理云台测试板可以填写：

```text
PTZ_CONTROLLER=virtual
VIRTUAL_PTZ_IMAGE=/home/glr/prj/yolo_detection/data/105.jpg
```

该文件包含现场配置，已被 `.gitignore` 忽略，不应提交真实账号或密码。

### 18.3 systemd 验证

编译后必须重启服务，确认运行的不是旧 Python 进程：

```bash
sudo systemctl restart ai_detection.service
systemctl status ai_detection.service --no-pager
curl -fsS http://127.0.0.1:3208/health
```

使用 root 权限检查运行进程实际加载的扩展模块：

```bash
PID="$(systemctl show -p MainPID --value ai_detection.service)"
sudo grep -E 'recognition_http_server/(service|virtual_ptz|dial_reading/pipeline).*\.so' "/proc/$PID/maps"
```

使用两张量程为 `25` 的表计图片执行 HTTP 回归：

```bash
/home/glr/miniconda3/envs/yolo-jetson/bin/python test_http.py \
  --server http://127.0.0.1:3208 \
  --images data/105.jpg data/156.jpg \
  --data-type 1:25 \
  --timeout 180
```
