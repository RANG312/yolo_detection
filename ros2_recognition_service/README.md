# ros2_recognition_service

`ros2_recognition_service` 是对根目录 `server.py` 的 ROS 2 封装版本。

它保留了原有识别能力：

- 表计读数
- 火源检测
- 安全帽检测

同时把原先的 HTTP 接口改成了：

- 一个提交任务的 ROS 2 service
- 一个查询任务状态的 ROS 2 service
- 一个发布异步结果的 ROS 2 topic
- 一个可选的图像 topic 订阅入口

如果之前没用过 ROS 2 service，可以直接按本文档的“编译 -> 启动 -> 调用 -> 看回调”顺序操作。

## 1. 目录结构

```text
ros2_recognition_service/
├── CMakeLists.txt
├── config/
│   └── recognition.yaml
├── launch/
│   └── recognition.launch.py
├── package.xml
├── srv/
│   ├── GetRecognitionTask.srv
│   └── SubmitRecognitionTask.srv
└── ros2_recognition_service/
    ├── __init__.py
    └── node.py
```

其中：

- `node.py` 是 ROS 2 节点入口
- `config/recognition.yaml` 是推荐修改的启动配置文件
- `deploy_ros2_ws.sh` 可从当前 `ultralytics` 源码目录自动创建并编译 ROS 2 workspace
- `source_ros2_ws.bash` 是仓库内的环境加载脚本
- `launch_recognition_service.sh` 是仓库内的启动脚本
- `launch/recognition.launch.py` 是推荐使用的启动入口
- `SubmitRecognitionTask.srv` 用于提交任务
- `GetRecognitionTask.srv` 用于按 `req_id` 查询任务状态

默认约定：

- 编译 ROS 2 包时使用系统 Python
- 启动识别服务时自动激活 conda 环境 `yolo-jetson`

## 2. 工作流程

这个节点是异步处理模型推理的，流程和原 HTTP 服务基本一致：

1. 客户端调用提交 service，发送识别请求
2. 节点立即返回“任务已接收”，并生成/记录 `req_id`
3. 后台线程执行识别
4. 识别完成后，把结果 JSON 发布到 callback topic
5. 客户端如果需要，也可以继续调用查询 service 查看任务状态和结果

如果配置了图像订阅 topic，还支持另一条触发路径：

1. 节点订阅 `sensor_msgs/msg/Image`
2. 收到图像后先落盘到本地临时目录
3. 节点自动创建一个内部识别任务
4. 后续识别、状态管理、回调 topic 发布，仍复用同一套流程

默认接口名：

- 提交 service：`/api/v1/recognition/tasks/submit`
- 查询 service：`/api/v1/recognition/tasks/get`
- 回调 topic：`/api/v1/recognition/callback`

## 3. 接口定义

### 3.1 提交任务 service

接口文件：`srv/SubmitRecognitionTask.srv`

```srv
string request_json
---
string response_json
```

也就是说，这个 service 的请求和响应都只包了一层字符串：

- 请求里传 `request_json`
- 返回里拿 `response_json`

这样做是为了最大限度复用 `server.py` 现有业务 JSON 协议，不需要第一次封装时就把所有字段拆成 ROS message。

### 3.2 查询状态 service

接口文件：`srv/GetRecognitionTask.srv`

```srv
string req_id
---
string response_json
```

调用时只需要传入任务 ID。

### 3.3 回调 topic

回调 topic 类型是：

```text
std_msgs/msg/String
```

topic 中的 `data` 字段内容，是识别完成后的完整 JSON 字符串。

## 4. 请求 JSON 格式

提交任务时，`request_json` 的内容与原 HTTP 版本的请求体保持一致。

一个最小可用示例如下：

```json
{
  "req_id": "demo-0001",
  "image_path": "/data/test/meter.jpg",
  "data_type": [
    {
      "recognize_type": "1",
      "recognize_subtype": "default"
    }
  ],
  "extra_info": "{\"debug_center\": false}"
}
```

常用字段说明：

- `req_id`：任务 ID，可传可不传；不传会自动生成
- `image_path`：图片路径，支持本地路径或 `http/https` URL；多张图时用英文逗号拼接
- `data_type`：识别项列表
- `extra_info`：额外参数，支持对象或 JSON 字符串

`recognize_type` 当前支持：

- `1`：表计读数
- `6`：火源检测
- `7`：安全帽检测

也兼容这些别名：

- `meter` / `dict_meter_type` -> `1`
- `fire` -> `6`
- `safehat` / `person` -> `7`

多图规则与 HTTP 版本相同：

- 如果 `data_type` 只有 1 项，则复用到所有图片
- 如果 `data_type` 有多项，则数量必须与图片数一致，并按顺序一一对应

## 5. 返回 JSON 格式

### 5.1 提交任务返回

提交 service 成功后，`response_json` 类似：

```json
{
  "req_id": "demo-0001",
  "code": 0,
  "resp_msg": "Task accepted successfully.",
  "data": {
    "task_status": "processing"
  }
}
```

### 5.2 查询任务返回

查询 service 返回的 `response_json` 类似：

```json
{
  "req_id": "demo-0001",
  "code": 0,
  "resp_msg": "success",
  "data": {
    "task_status": "finished",
    "created_at": "2026-04-13 10:00:00",
    "updated_at": "2026-04-13 10:00:05",
    "callback_payload": {
      "req_id": "demo-0001",
      "data_result": [],
      "code": 0,
      "resp_msg": "Task finished successfully.",
      "finish_time": "2026-04-13 10:00:05"
    },
    "callback_status_code": 0,
    "callback_error": null,
    "error": null
  }
}
```

如果任务不存在，会返回错误 JSON。

### 5.3 回调 topic 消息

识别完成后，topic 里会发布字符串消息，内容类似：

```json
{
  "req_id": "demo-0001",
  "data_result": [
    {
      "image_path": "/data/test/meter.jpg",
      "image_path_result": "/data/test/meter-detection.jpg",
      "recognize_data": []
    }
  ],
  "code": 0,
  "resp_msg": "Task finished successfully.",
  "finish_time": "2026-04-13 10:00:05"
}
```

## 6. 环境准备

这个 ROS 2 包本身只是一层封装，实际推理仍依赖根目录的 `server.py`、模型权重和 Python 依赖。

因此运行前至少需要满足：

- 已安装 ROS 2
- 已安装 `colcon`
- 已安装 ROS 2 Python 依赖：`rclpy`、`sensor_msgs`、`std_msgs`、`cv_bridge`、`rosidl_default_generators`、`rosidl_default_runtime`
- 当前工程中的 Python 推理依赖可正常使用
- 模型文件路径可访问

另外，`node.py` 会导入仓库根目录的 `server.py`。

默认仓库根路径写死为：

```text
/data/prj/yolov8_dial_reading/ultralytics
```

如果你的仓库不在这个路径下，启动前请设置：

```bash
export RECOGNITION_REPO_ROOT=/你的/ultralytics/仓库路径
```

## 7. 编译

如果客户机器上拿到的是完整 `ultralytics` 源码目录，推荐优先使用自动部署脚本：

```bash
cd /你的/ultralytics/仓库路径/ros2_recognition_service
bash deploy_ros2_ws.sh
```

脚本会自动完成：

- 创建 ROS 2 workspace
- 挂载 `ros2_recognition_service` 包
- 生成一份带绝对路径的启动配置文件
- 生成工作区级环境脚本 `source_ros2_ws.bash`
- 生成工作区级启动脚本 `launch_recognition_service.sh`
- 清理旧的构建缓存
- 强制使用系统 Python 编译 ROS 2 包，避免 conda 缓存污染
- 执行 `colcon build`

脚本执行完后，会打印后续 `source` 和 `ros2 launch` 命令。

推荐直接使用脚本生成的两个文件：

- `~/prj/ros2_ws/source_ros2_ws.bash`
- `~/prj/ros2_ws/launch_recognition_service.sh`

仓库中也保留了同名脚本，方便一起提交到 GitLab：

- `ros2_recognition_service/source_ros2_ws.bash`
- `ros2_recognition_service/launch_recognition_service.sh`

这两个脚本默认会：

- 先激活 conda 环境 `yolo-jetson`
- 再 `source /opt/ros/<发行版>/setup.bash`
- 再 `source ~/ros2_ws/install/setup.bash`
- 如果 workspace 顶层 setup 没把包叠加进环境，会自动 fallback 到包级 `local_setup.bash`

如果不想自动激活 conda，可以在执行前设置：

```bash
USE_CONDA_ENV=0 ./launch_recognition_service.sh
```

如果 conda 环境名不是 `yolo-jetson`，可以覆盖：

```bash
CONDA_ENV_NAME=你的环境名 ./launch_recognition_service.sh
```

如果需要自定义 workspace 或订阅话题，可以例如：

```bash
bash deploy_ros2_ws.sh \
  --workspace ~/prj/ros2_ws \
  --ros-distro humble \
  --build-python /usr/bin/python3 \
  --device cpu \
  --image-topic /camera/image_raw \
  --image-recognize-type 6
```

如果你在 Jetson 上同时使用 conda 环境做推理，建议遵循这个规则：

- 编译 ROS 2 包时使用系统 Python，例如 `/usr/bin/python3`
- 运行识别服务时激活 conda 环境，例如 `yolo-jetson`

如果你需要手动编译，也可以按下面步骤执行。

假设你已经有一个 ROS 2 workspace，例如：

```text
~/ros2_ws/
└── src/
    └── ros2_recognition_service/
```

把这个目录放进 workspace 的 `src/` 下，然后执行：

```bash
cd ~/ros2_ws
colcon build --packages-select ros2_recognition_service
```

编译完成后 source 环境：

```bash
source /opt/ros/ < 你的发行版 > /setup.bash
source ~/ros2_ws/install/setup.bash
```

如果你使用自动部署脚本，更推荐直接：

```bash
source ~/ros2_ws/source_ros2_ws.bash
```

## 8. 启动节点

推荐使用 `launch` 启动。

### 8.1 用 launch 启动

推荐方式是先改配置文件，再启动。

默认配置文件是：

```text
ros2_recognition_service/config/recognition.yaml
```

你可以先把里面这些字段改好：

- `device`
- `model`
- `fire_model`
- `safehat_model`
- `image_topic`
- `image_recognize_type`
- `image_recognize_subtype`

然后直接启动：

```bash
ros2 launch ros2_recognition_service recognition.launch.py
```

但如果你是通过 `deploy_ros2_ws.sh` 部署出来的 workspace，更推荐直接使用工作区级启动脚本：

```bash
~/ros2_ws/launch_recognition_service.sh
```

这个脚本会自动：

- `source /opt/ros/<发行版>/setup.bash`
- `source ~/ros2_ws/install/setup.bash`
- 带上 `config_file:=~/ros2_ws/config/recognition.yaml`

这样可以避免误读到包安装目录里的默认配置文件。

### 8.2 指定配置文件启动

如果你想准备多套配置，可以通过 `config_file:=...` 指定：

```bash
ros2 launch ros2_recognition_service recognition.launch.py \
  config_file:=/你的路径/recognition.yaml
```

注意：

- 如果不显式传 `config_file:=...`，`launch` 会读取包安装目录里的默认 `config/recognition.yaml`
- 自动部署脚本生成在 workspace 下的 `~/ros2_ws/config/recognition.yaml` 不会自动生效
- 所以客户侧推荐直接使用 `~/ros2_ws/launch_recognition_service.sh`

### 8.3 命令行临时覆盖参数

如果只想临时改一两个值，也可以在 launch 命令后直接覆盖，例如：

```bash
ros2 launch ros2_recognition_service recognition.launch.py \
  device:=cpu \
  image_topic:=/camera/image_raw
```

命令行覆盖优先级高于 YAML 配置文件。

如果仓库根路径不是默认值，也可以一起传：

```bash
ros2 launch ros2_recognition_service recognition.launch.py \
  repo_root:=/你的/ultralytics/仓库路径 \
  device:=cpu
```

常用 launch 参数：

- `repo_root`：仓库根路径，会写入环境变量 `RECOGNITION_REPO_ROOT`
- `node_name`：节点名，默认 `recognition_service_node`
- `submit_service`：提交 service 名
- `status_service`：查询 service 名
- `callback_topic`：结果 topic 名
- `image_topic`：可选的图像订阅 topic，消息类型为 `sensor_msgs/msg/Image`
- `image_qos_depth`：图像订阅队列深度
- `image_recognize_type`：订阅图像生成任务时使用的 `recognize_type`
- `image_recognize_subtype`：订阅图像生成任务时使用的 `recognize_subtype`
- `device`：推理设备，例如 `cpu` 或 `0`
- `result_root`：输入输出目录根路径

### 8.4 用 ros2 run 直接启动

如果只是临时调试，也可以继续直接启动节点：

```bash
ros2 run ros2_recognition_service recognition_node -- \
  --device cpu \
  --model runs/train/meter_data_9k_yolov8m_best/weights/best.pt \
  --fire-model runs/train/best_fire.pt \
  --safehat-model runs/train/best_person.pt \
  --result-root results/http_service
```

常用节点参数：

- `--node-name`：节点名，默认 `recognition_service_node`
- `--submit-service`：提交 service 名
- `--status-service`：查询 service 名
- `--callback-topic`：结果 topic 名
- `--device`：推理设备，例如 `cpu` 或 `0`
- `--result-root`：输入输出目录根路径

节点启动后会打印：

- 三个模型加载路径
- 表计标注模式 `annotation_mode`
- service/topic 名称
- 如果启用了图像订阅，也会打印订阅的图片 topic 和默认检测类型

### 8.5 用订阅图像 topic 触发识别

如果你想让节点直接消费相机图像，而不是手工调用提交 service，可以在 launch 时配置：

```bash
ros2 launch ros2_recognition_service recognition.launch.py \
  device:=cpu \
  image_topic:=/camera/image_raw \
  image_recognize_type:=6 \
  image_recognize_subtype:=
```

这表示：

- 订阅 `/camera/image_raw`
- 每收到一张 `sensor_msgs/msg/Image`
- 就自动创建一个识别任务
- 该任务按 `recognize_type=6` 也就是火源检测执行

如果要做表计读数，可以例如：

```bash
ros2 launch ros2_recognition_service recognition.launch.py \
  device:=cpu \
  image_topic:=/camera/image_raw \
  image_recognize_type:=1 \
  image_recognize_subtype:=default
```

订阅得到的图像会先保存到：

```text
<result_root>/inputs/ros2_topic/<req_id>/frame.jpg
```

然后复用现有识别任务流程，所以结果仍然会：

- 出现在查询 service 中
- 发布到 callback topic 中

## 9. 第一次使用时怎么调

如果你之前没用过 ROS 2 service，建议开 3 个终端。

### 9.1 终端 1：启动节点

```bash
source /opt/ros/ < 你的发行版 > /setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch ros2_recognition_service recognition.launch.py device:=cpu
```

### 9.2 终端 2：订阅回调 topic

```bash
source /opt/ros/ < 你的发行版 > /setup.bash
source ~/ros2_ws/install/setup.bash
ros2 topic echo /api/v1/recognition/callback
```

只要任务完成，这个终端就会收到一条 `std_msgs/msg/String` 消息。

### 9.3 终端 3：提交任务

先看一下 service 是否已经起来：

```bash
ros2 service list | grep recognition
```

再调用提交接口：

```bash
ros2 service call /api/v1/recognition/tasks/submit ros2_recognition_service/srv/SubmitRecognitionTask \
  '{request_json: "{\"req_id\":\"demo-0001\",\"image_path\":\"/data/test/meter.jpg\",\"data_type\":[{\"recognize_type\":\"1\",\"recognize_subtype\":\"default\"}],\"extra_info\":\"{\\\"debug_center\\\": false}\"}"}'
```

如果请求合法，会立即返回一个 `response_json`，其中 `task_status` 通常是 `processing`。

### 9.4 查询任务状态

```bash
ros2 service call /api/v1/recognition/tasks/get ros2_recognition_service/srv/GetRecognitionTask \
  '{req_id: "demo-0001"}'
```

任务完成后，返回的 `response_json` 里可以看到：

- `task_status`
- `callback_payload`
- `callback_status_code`
- `callback_error`
- `error`

## 10. 常用排查方法

### 10.1 看接口定义

```bash
ros2 interface show ros2_recognition_service/srv/SubmitRecognitionTask
ros2 interface show ros2_recognition_service/srv/GetRecognitionTask
```

### 10.2 看节点、service、topic

```bash
ros2 node list
ros2 service list | grep recognition
ros2 topic list | grep recognition
```

### 10.3 看 topic 类型

```bash
ros2 topic info /api/v1/recognition/callback
```

### 10.4 常见问题

`ros2 run` 找不到包：

- 通常是没有重新 `colcon build`
- 或者忘了 `source ~/ros2_ws/install/setup.bash`

`ros2 launch` 找不到 launch 文件：

- 先确认 `launch/recognition.launch.py` 已经安装到包里
- 再确认修改后重新执行过 `colcon build`

`ros2 launch` 提示配置文件不存在或格式错误：

- 先确认 `config_file` 路径存在
- 配置文件内容必须是标准 YAML
- 顶层必须是键值对结构，不能写成列表

service 调用时报 JSON 解析错误：

- 先确认传入的是合法 JSON 字符串
- 再确认 shell 转义是否正确

节点启动时报模型或路径错误：

- 检查 `RECOGNITION_REPO_ROOT`
- 检查权重路径是否存在
- 检查当前 Python 环境是否能正常导入 `server.py` 依赖

订阅图像没有触发任务：

- 先确认 `image_topic` 不是空
- 再确认上游发布的话题类型确实是 `sensor_msgs/msg/Image`
- 用 `ros2 topic echo <你的图像话题>` 或 `ros2 topic info <你的图像话题>` 检查是否有数据

## 11. 实现说明

这个 ROS 2 版本没有改动核心识别逻辑，只是替换了传输层：

- 复用 `server.py` 中的 `RecognitionService`
- 后台仍然是异步线程处理任务
- 原本的 HTTP 回调，改成向 ROS 2 topic 发布 JSON 字符串

因此业务行为应尽量和原 HTTP 服务保持一致，便于已有调用方迁移。
