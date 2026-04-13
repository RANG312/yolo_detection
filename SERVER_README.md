# HTTP识别服务说明

本文档说明当前项目中的两个脚本：

- `server.py`：识别服务端，提供任务提交、任务查询、结果回调能力
- `test_http.py`：本地联调用测试客户端，用于向 `server.py` 发请求并接收回调


## 1. 目标与整体流程

当前服务支持 3 类识别场景：

- 表计读数
- 火源检测
- 安全帽检测

整体调用流程如下：

1. 客户端向 `POST /api/v1/recognition/tasks` 提交识别任务。
2. 服务端立即返回受理结果，任务状态为 `processing`。
3. 服务端后台线程执行识别。
4. 识别完成后，服务端主动回调 `POST /api/v1/recognition/callback`。
5. 客户端也可以通过 `GET /api/v1/recognition/tasks/{req_id}` 查询任务状态和回调结果。


## 2. server.py 业务逻辑

### 2.1 服务职责

`server.py` 的职责是：

- 启动一个 HTTP 服务
- 接收识别任务
- 根据请求中的 `data_type` 选择对应模型
- 对图片执行识别
- 保存输入图片和可视化结果
- 在识别完成后回调结果
- 提供任务状态查询接口


### 2.2 当前加载的模型

`server.py` 启动时会一次性加载 3 个模型：

- 表计模型：默认 `runs/train/meter_data_9k_yolov8m_best/weights/best.pt`
- 火源模型：默认 `runs/train/best_fire.pt`
- 安全帽模型：默认 `runs/train/best_person.pt`

对应启动参数：

- `--model`
- `--fire-model`
- `--safehat-model`


### 2.3 识别类型分发规则

服务通过请求体中的 `data_type` 决定执行哪类识别。

当前约定：

- `recognize_type = "1"`：表计读数
- `recognize_type = "6"`：火源检测
- `recognize_type = "7"`：安全帽检测

代码里还兼容了一些历史别名：

- `"dict_meter_type"` 会被当成表计
- `"fire"` 会被当成火源检测
- `"safehat"` 或 `"person"` 会被当成安全帽检测


### 2.4 请求体关键字段

提交任务接口：

- 路径：`/api/v1/recognition/tasks`
- 方法：`POST`

请求体示例：

```json
{
  "req_id": "9d5e6c7a-0001-0002-0003-000000000001",
  "image_path": "/data/test/aaa.jpg,/data/test/bbb.jpg",
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

- `req_id`：任务ID，可传可不传；不传时服务端自动生成
- `image_path`：图片路径，支持本地路径或 HTTP URL；多张图片使用英文逗号分隔
- `data_type`：识别项列表
- `extra_info`：额外信息，支持 JSON 字符串或对象


### 2.5 data_type 的处理规则

`data_type` 是一个数组，每个元素描述一种识别需求：

```json
[
  {"recognize_type": "1", "recognize_subtype": "3"},
  {"recognize_type": "6", "recognize_subtype": ""}
]
```

处理规则如下：

- 如果未传 `data_type`，默认按表计处理，等价于：

```json
[
  {"recognize_type": "1", "recognize_subtype": "default"}
]
```

- 表计场景允许 `recognize_subtype` 为空；为空时自动补成 `"default"`
- 火源检测和安全帽检测没有 subtype 概念，因此允许 `recognize_subtype` 为空字符串
- 当前阶段不对表计 subtype 做业务校验，所有 `dict_meter_type` 都允许通过


### 2.6 多图片处理

`image_path` 支持多张图片，例如：

```text
/data/test/1.jpg,/data/test/2.jpg,https://example.com/3.jpg
```

服务端会：

1. 按逗号拆分图片路径
2. 逐张处理
3. 将每张图的结果写入 `data_result`


### 2.7 本地图片与URL图片处理

服务端对图片路径的处理分两类：

- 如果是 `http://` 或 `https://`，会先下载到 `results/http_service/inputs/<req_id>/`
- 如果是本地路径，则直接读取

输出结果图统一保存在：

```text
results/http_service/outputs/<req_id>/
```


### 2.8 三类识别的具体执行逻辑

#### 2.8.1 表计读数

表计场景走的是 `dial_reading.py` 中的专用逻辑，而不是简单目标检测。

执行步骤：

1. 使用表计 YOLO 模型检测表盘关键点或关键框
2. 调用 `predict_image_instances(...)`
3. 在几何后处理中计算指针位置
4. 得到归一化读数 `reading`
5. 生成可视化图片

表计结果中的：

- `recognize_value`：归一化后的读数字符串
- `recognize_desc`：包含表计编号、读数和 `arc_mode`

说明：

- 这里返回的是归一化读数，不是物理量换算值
- 代码中 `DEFAULT_MIN_VALUE=0.0`、`DEFAULT_MAX_VALUE=1.0`，所以当前返回区间是 `0~1`


#### 2.8.2 火源检测

火源检测走通用 YOLO 检测逻辑。

执行步骤：

1. 使用 `runs/train/best_fire.pt`
2. 对整图执行 `model.predict(...)`
3. 将检测框绘制到可视化图上
4. 遍历每个检测框生成返回项

火源数据集类别来自 `runs/train/fire.yaml`：

- `fire`
- `smoke`

返回时：

- `recognize_value`：类别名，例如 `fire`、`smoke`
- `confidence`：置信度百分比整数，例如 `87`

如果没有检测到目标，则返回一条失败记录：

- `recognize_image_index = "0"`
- `recognize_value = ""`
- `confidence = "0"`


#### 2.8.3 安全帽检测

安全帽检测同样走通用 YOLO 检测逻辑。

执行步骤：

1. 使用 `runs/train/best_person.pt`
2. 对整图执行 `model.predict(...)`
3. 将检测结果写入可视化图
4. 每个检测框输出一条识别结果

类别名来自 `runs/train/safehat.yaml`，例如：

- `Hardhat`
- `NO-Hardhat`
- `Person`
- `Safety Vest`
- `machinery`
- `vehicle`

返回时：

- `recognize_value`：检测类别名
- `confidence`：置信度百分比整数


### 2.9 多张图片与检测项的对应关系

当前业务规则是：一张图只做一种检测。

如果一个请求里有多张图片，则有两种写法。

第一种：只给 1 个 `data_type`，表示所有图片都做同一种检测，例如：

```json
[
  {"recognize_type": "6", "recognize_subtype": ""}
]
```

如果请求中有：

```json
{
  "image_path": "a.jpg,b.jpg,c.jpg",
  "data_type": [
    {"recognize_type": "6", "recognize_subtype": ""}
  ]
}
```

则等价于：

- `a.jpg` 做火源检测
- `b.jpg` 做火源检测
- `c.jpg` 做火源检测

第二种：给多个 `data_type`，则必须与图片数量一致，并且按顺序一一对应，例如：

```json
{
  "image_path": "a.jpg,b.jpg,c.jpg",
  "data_type": [
    {"recognize_type": "1", "recognize_subtype": "3"},
    {"recognize_type": "6", "recognize_subtype": ""},
    {"recognize_type": "7", "recognize_subtype": ""}
  ]
}
```

其对应关系是：

- `a.jpg` -> 表计读数
- `b.jpg` -> 火源检测
- `c.jpg` -> 安全帽检测

如果 `image_path` 数量和 `data_type` 数量都大于 1，但两者不相等，服务端会直接报错。

注意：

- 当前不会对同一张图叠加执行多种识别
- `recognize_data` 只描述当前图片所属检测项的结果
- `image_path_result` 是当前图片对应检测项的结果图路径


### 2.10 返回结果结构

回调结果顶层结构：

```json
{
  "req_id": "xxx",
  "data_result": [
    {
      "image_path": "/data/test/a.jpg",
      "image_path_result": "results/http_service/outputs/<req_id>/a_result_fire.jpg",
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
  "finish_time": "2026-04-08 14:10:00"
}
```


### 2.11 recognize_data 字段含义

每一项都包含以下字段：

- `recognize_image_index`
- `recognize_type`
- `recognize_subtype`
- `recognize_value`
- `confidence`
- `recognize_desc`

字段解释：

- `recognize_image_index`
  - 表计场景：对应第几个表计实例
  - 检测场景：对应第几个检测框
  - 未检出目标时固定为 `"0"`
- `recognize_type`
  - 当前请求中的识别类型
- `recognize_subtype`
  - 表计时可为业务 subtype
  - 火源/安全帽通常为空字符串
- `recognize_value`
  - 表计：归一化读数
  - 火源/安全帽：检测类别名
- `confidence`
  - 表计当前固定为 `"100"`
  - 检测场景按模型置信度转换为百分比整数
- `recognize_desc`
  - 面向业务侧的结果说明


### 2.12 任务状态查询接口

查询接口：

- 路径：`/api/v1/recognition/tasks/{req_id}`
- 方法：`GET`

返回内容包含：

- `task_status`
- `created_at`
- `updated_at`
- `callback_payload`
- `callback_status_code`
- `callback_error`
- `error`

用途：

- 观察后台任务是否完成
- 查看服务端最终回调的数据
- 在回调失败时排查问题


### 2.13 健康检查接口

健康检查：

- 路径：`/health`
- 方法：`GET`

可用于容器探活或进程监控。


## 3. test_http.py 业务逻辑

### 3.1 脚本职责

`test_http.py` 是一个联调脚本，用于：

- 启动本地回调服务
- 向 `server.py` 提交任务
- 等待服务端回调
- 校验返回结构
- 将回调结果保存为 JSON 文件
- 在终端打印摘要


### 3.2 它解决了什么问题

因为 `server.py` 的识别结果是异步回调的，所以单纯 `curl` 提交任务不够方便。

`test_http.py` 的作用是一次性完成：

1. 启动回调监听
2. 发任务
3. 收回调
4. 打印结果

适合开发联调和本地验证。


### 3.3 回调服务器逻辑

`test_http.py` 内部会启动一个本地 HTTP 服务：

- 默认监听：`0.0.0.0:18080`
- 默认路径：`/api/v1/recognition/callback`

当 `server.py` 完成识别后，会把回调结果 POST 到这里。


### 3.4 请求构造逻辑

`test_http.py` 会把所有 `--images` 参数用逗号拼接为 `image_path`。

示例：

```bash
--images /data/a.jpg /data/b.jpg
```

会生成：

```json
"image_path": "/data/a.jpg,/data/b.jpg"
```


### 3.5 scene 预设逻辑

为了方便测试，`test_http.py` 新增了 `--scene`：

- `meter`
- `fire`
- `safehat`

当没有显式传 `--data-type` 时，会根据 `--scene` 自动补默认值。

默认映射：

- `meter`

```json
[
  {"recognize_type": "1", "recognize_subtype": "3"},
  {"recognize_type": "1", "recognize_subtype": "5"}
]
```

- `fire`

```json
[
  {"recognize_type": "6", "recognize_subtype": ""}
]
```

- `safehat`

```json
[
  {"recognize_type": "7", "recognize_subtype": ""}
]
```


### 3.6 自定义 data_type 逻辑

如果显式传了 `--data-type`，则优先使用显式值，不再使用 `--scene` 默认值。

格式如下：

```text
type[:subtype]
```

示例：

- `1:3`
- `1:5`
- `6`
- `7`
- `meter:default`
- `fire`
- `safehat`

脚本内部会做标准化：

- `meter` -> `1`
- `dict_meter_type` -> `1`
- `fire` -> `6`
- `safehat` -> `7`
- `person` -> `7`

其中：

- 表计如果不写 subtype，会自动补为 `default`
- 火源和安全帽允许 subtype 为空


### 3.7 回调结果校验逻辑

收到回调后，`test_http.py` 会校验返回结构是否完整。

顶层必须有：

- `req_id`
- `data_result`
- `code`
- `resp_msg`
- `finish_time`

每个 `recognize_data` 项必须有：

- `recognize_image_index`
- `recognize_type`
- `recognize_subtype`
- `recognize_value`
- `confidence`
- `recognize_desc`


### 3.8 结果保存与摘要打印

收到回调后，脚本会：

1. 把完整 JSON 保存到默认文件
2. 在终端打印摘要

默认输出文件：

```text
results/http_service/test_callback_payload.json
```

摘要中会显示：

- `recognize_image_index`
- `type`
- `subtype`
- `value`
- `confidence`
- `desc`


## 4. 启动与测试示例

### 4.1 启动服务

```bash
python3 server.py
```

如果需要显式指定模型：

```bash
python3 server.py \
  --model runs/train/meter_data_9k_yolov8m_best/weights/best.pt \
  --fire-model runs/train/best_fire.pt \
  --safehat-model runs/train/best_person.pt
```


### 4.2 测试表计

```bash
python3 test_http.py \
  --images /path/to/meter.jpg \
  --scene meter
```


### 4.3 测试火源检测

```bash
python3 test_http.py \
  --images /path/to/fire.jpg \
  --scene fire
```


### 4.4 测试安全帽检测

```bash
python3 test_http.py \
  --images /path/to/safehat.jpg \
  --scene safehat
```


### 4.5 多图按顺序对应不同检测项

下面这个请求表示 3 张图分别做 3 种不同检测：

```bash
python3 test_http.py \
  --images /path/to/meter.jpg /path/to/fire.jpg /path/to/safehat.jpg \
  --data-type 1:3 \
  --data-type 6 \
  --data-type 7
```


## 5. 结果文件目录

默认目录结构：

```text
results/http_service/
├── inputs/
│   └── <req_id>/
│       └── 下载的URL图片
├── outputs/
│   └── <req_id>/
│       ├── xxx_result_meter.jpg
│       ├── xxx_result_fire.jpg
│       └── xxx_result_safehat.jpg
└── test_callback_payload.json
```


## 6. 当前实现的几个注意点

### 6.1 表计返回的是归一化读数

当前表计读数不是业务量程值，而是按 `0~1` 的归一化结果返回。

如果后续要按不同表计 subtype 映射成真实量程，需要再加一层业务换算逻辑。


### 6.2 多图混合场景依赖顺序对应

如果一个请求里有多张图片且 `data_type` 传了多项，那么当前逻辑严格按顺序绑定：

- 第 1 张图对应第 1 个 `data_type`
- 第 2 张图对应第 2 个 `data_type`
- 以此类推

因此调用方必须自己保证顺序正确。


### 6.3 表计 confidence 当前固定为100

表计走的是几何读数逻辑，不是直接返回单个检测框概率，因此目前：

- `confidence = "100"`

如果后续希望更合理，需要定义表计结果置信度计算方式。


### 6.4 服务端回调地址来源

`server.py` 当前会根据请求来源 IP 组装回调地址：

```text
http://<callback_host>:18080/api/v1/recognition/callback
```

也就是说：

- 服务端并不会直接读取 `extra_info.callback_url`
- 本地联调时要确保客户端所在机器能被服务端访问到 `18080` 端口


## 7. 建议的后续优化

- 为表计 subtype 增加真实量程映射
- 为火源和安全帽增加更明确的业务字段定义
- 支持多模型结果叠加输出到同一张结果图
- 支持直接从请求中读取显式 callback URL
- 为 `test_http.py` 增加示例图片和批量测试能力
