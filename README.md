# HTTP 识别服务说明

本文档说明当前仓库中的 `HTTP` 识别服务、测试脚本和代码结构。

相关文件：

- [server.py](/data/prj/yolov8_dial_reading/ultralytics/server.py)：兼容入口层，保留原有导出，供外部脚本和 ROS2 代码继续使用
- [recognition_http_server/app.py](/data/prj/yolov8_dial_reading/ultralytics/recognition_http_server/app.py)：HTTP 服务启动入口
- [recognition_http_server/service.py](/data/prj/yolov8_dial_reading/ultralytics/recognition_http_server/service.py)：核心任务编排层
- [test_http.py](/data/prj/yolov8_dial_reading/ultralytics/test_http.py)：本地联调脚本

## 快速开始

### http server
- conda环境部署：运行 `bash deploy_yolo_jetson_env.sh`自动部署环境，同时会删除相关的安装包和.git节省空间, 并添加自启的`systemd`服务
- 日志：` results/http_service/logs/server.log `
- 修改参数： ` recognition_http_server/constants.py ` 

### ROS2 humble
- 先运行 ``` bash ros2_recognition_service/deploy_ros2_ws.sh``` 构建ros2工作区并编译
- 再运行 ``` source ./ros2_recognition_service/source_ros2_ws.bash ```
- 接着ros2 launch 节点 ``` bash ros2_recognition_service/launch_recognition_service.sh ```
- 可以在 `ros2_recognition_service/config/recognition.yaml` 中设置相关配置参数
- 详细介绍参考 `ros2_recognition_service/README.md `

## 1. 总体说明

当前服务支持 3 类识别任务：

- 表计读数
- 火源检测
- 安全帽检测

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
- `handlers/detection.py`：通用检测执行链路
- `handlers/meter.py`：表计相关执行链路
- `utils/image_io.py`：图片下载和本地路径准备

兼容性说明：

- `server.py` 当前仍然保留
- `server.py` 会重新导出 `RecognitionService`、`TaskState`、`make_error_payload`、`now_text` 和各类 `DEFAULT_*` 常量
- 这样做是为了不影响当前 ROS2 服务对 `server.py` 的依赖

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
- `--imgsz`：推理尺寸
- `--conf`：置信度阈值
- `--device`：推理设备，例如 `cpu`、`0`
- `--min-value`：表计默认最小值
- `--max-value`：表计默认最大值
- `--result-root`：结果目录，默认 `results/http_service`
- `--callback-port`：回调端口，默认 `8088`
- `--request-timeout`：图片下载和回调超时时间

示例：

```bash
python server.py \
  --host 0.0.0.0 \
  --port 3208 \
  --model runs/train/meter_data_9k_yolov8m_best/weights/best.pt \
  --fire-model runs/train/best_fire.pt \
  --safehat-model runs/train/best_person.pt \
  --device 0
```

## 5. 当前模型加载

服务启动时会一次性加载 3 个模型：

- 表计模型：默认 `runs/train/meter_data_9k_yolov8m_best/weights/best.pt`
- 火源模型：默认 `runs/train/best_fire.pt`
- 安全帽模型：默认 `runs/train/best_person.pt`

说明：

- 表计任务当前仍然复用 [dial_reading.py](/data/prj/yolov8_dial_reading/ultralytics/dial_reading.py) 的关键点检测和几何后处理逻辑
- 火源和安全帽任务走通用 YOLO 检测链路

## 6. 任务类型路由规则

服务通过请求中的 `data_type[].recognize_type` 决定大类任务。

当前映射：

- `1`：表计
- `6`：火源检测
- `7`：安全帽检测

兼容别名：

- `dict_meter_type` -> `1`
- `fire` -> `6`
- `safehat` -> `7`
- `person` -> `7`

内部路由：

- `recognize_type` 先映射成内部 `task_kind`
- `task_kind` 再由 `RecognitionService` 中的 handler 注册表分发到具体处理函数

当前内置 handler：

- `meter`
- `fire`
- `safehat`

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
2. 调用 `dial_reading.py` 中的后处理逻辑
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

### 8.2 数码表

数码表链路目前只完成了服务端骨架，尚未真正接入 ROI 检测和 OCR。

当前行为：

- 当 `recognize_subtype="digital"` 时，会进入数码表处理入口
- 当前会返回一条失败结果，提示 OCR 链路尚未接入
- 同时保存原图到结果图路径，方便后续联调

这部分是为后续接入 `ROI 检测 -> OCR` 链路预留的扩展口

## 9. 纯检测任务当前行为

火源检测和安全帽检测都复用通用检测执行器。

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

### 15.3 参数传法说明

`test_http.py` 当前支持两种传法：

第一种：紧凑写法，使用 `--data-type type[:subtype]`

- `--data-type 1:3`
- `--data-type 1:25`
- `--data-type 1:digital`
- `--data-type 6`
- `--data-type 7`
- `--data-type meter:default`
- `--data-type fire`
- `--data-type safehat`

第二种：拆分写法，显式使用 `--recognize-type` 和 `--recognize-subtype`

- `--recognize-type 1 --recognize-subtype 25`
- `--recognize-type 1 --recognize-subtype digital`
- `--recognize-type 6`
- `--recognize-type fire`
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
