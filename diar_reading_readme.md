# dial_reading.py 说明

`dial_reading.py` 用于对单张表盘图像执行目标检测，并基于检测点位计算表盘读数。

当前脚本默认适配新数据集 `data/new_datas/meter_data_9k`，默认权重为：

- `runs/train/meter_data_9k_yolov8m_20260407_113957/weights/best.pt`

脚本仍兼容旧版三类模型，但当前主路径已经切换到 `meter_data_9k` 的 5 类标注逻辑。

## 当前默认类别

`meter_data_9k` 模型类别必须包含：

- `gauge`
- `center`
- `pointer_tip`
- `max_tick`
- `min_tick`

其中：

- `gauge` 表示整块表盘区域
- `center` 表示表盘中心点的小框
- `pointer_tip` 表示指针尖端区域的小框
- `min_tick` 表示起始刻线点的小框
- `max_tick` 表示终止刻线点的小框

当前这套标注逻辑中，`center`、`pointer_tip`、`min_tick`、`max_tick` 都把检测框的几何中心当作真实点位。

## 功能流程

脚本整体流程如下：

1. 读取输入图像。
2. 加载 YOLO 模型，自动识别当前是 `meter_data_9k` 模式还是旧版 `legacy` 模式。
3. 对图像执行检测，并从每个类别中选取置信度最高的一个框。
4. 在 `meter_data_9k` 模式下，直接取 `center`、`pointer_tip`、`min_tick`、`max_tick` 的框中心作为几何点。
5. 将起点刻度、终点刻度、指针尖端转换为相对圆心的角度。
6. 根据角度关系计算读数比例，并映射到 `[min_value, max_value]`。
7. 输出预测值，并保存带可视化标注的结果图。

旧版 `legacy` 模式下，脚本仍保留原先基于 ROI 的圆心、刻线和指针几何估计逻辑。

## 命令行参数

脚本入口参数定义见 [dial_reading.py](/data/prj/yolov8_dial_reading/ultralytics/dial_reading.py#L18)。

常用参数如下：

- `--model`：模型路径，默认是 `runs/train/meter_data_9k_yolov8m_20260407_113957/weights/best.pt`
- `--image`：输入图片路径，单图模式必填
- `--imgsz`：推理尺寸，默认 `640`
- `--conf`：检测置信度阈值，默认 `0.25`
- `--device`：推理设备，默认 `"0"`
- `--min-value`：表盘最小值，默认 `0.0`
- `--max-value`：表盘最大值，默认 `25`
- `--test-loop`：批量处理 `--input-dir` 下所有图片
- `--input-dir`：批量测试图片目录，默认 `data/new_datas/meter_data_9k/images/val`
- `--debug-center`：输出圆心相关调试点
- `--show`：显示结果图
- `--save`：保存结果图路径，默认输出到 `results/dial_reading_<timestamp>.png`

## 使用示例

### 1. 使用默认 `meter_data_9k` 权重进行预测

```bash
python3 dial_reading.py \
  --image data/new_datas/meter_data_9k/images/val/000003_jpg.rf.8dabb48858838c2f71de1f1c3555563f.jpg \
  --device cpu
```

### 2. 指定模型与量程

```bash
python3 dial_reading.py \
  --model runs/train/meter_data_9k_yolov8m_20260407_113957/weights/best.pt \
  --image path/to/test.jpg \
  --min-value 0 \
  --max-value 100 \
  --device cpu
```

### 3. 批量处理验证集

```bash
python3 dial_reading.py \
  --test-loop \
  --input-dir data/new_datas/meter_data_9k/images/val \
  --device cpu
```

## 输出内容

脚本会在终端输出：

- 模型加载耗时
- 当前识别到的标注模式
- 模型推理耗时
- 读数计算耗时
- 量程范围
- 最终预测读数
- 圆弧方向模式
- 可视化结果图保存路径

## 可视化结果说明

在 `meter_data_9k` 模式下，输出图像会绘制以下内容：

- `gauge` 检测框
- `center` 检测框与点
- `min_tick` 检测框与点
- `max_tick` 检测框与点
- `pointer_tip` 检测框与点
- 圆心到上述点的连线
- 最终读数和圆弧模式文字

## 关键实现说明

### 1. 模式自动识别

脚本会根据模型类别自动判断当前使用：

- `meter_data_9k` 5 类模式
- `legacy` 3 类模式

逻辑见 [dial_reading.py](/data/prj/yolov8_dial_reading/ultralytics/dial_reading.py#L707)。

### 2. 检测结果筛选

每个类别只保留置信度最高的一个检测框，逻辑见 [dial_reading.py](/data/prj/yolov8_dial_reading/ultralytics/dial_reading.py#L682)。

### 3. `meter_data_9k` 点位计算

在新模式下：

- `center` 直接取框中心
- `min_tick` 直接取框中心
- `max_tick` 直接取框中心
- `pointer_tip` 直接取框中心

这部分逻辑见 [dial_reading.py](/data/prj/yolov8_dial_reading/ultralytics/dial_reading.py#L746)。

### 4. 读数计算

脚本将 `min_tick` 角、`max_tick` 角、`pointer_tip` 角转换为圆心角关系，再计算比例并映射到量程范围：

`reading = min_value + ratio * (max_value - min_value)`

相关逻辑见 [dial_reading.py](/data/prj/yolov8_dial_reading/ultralytics/dial_reading.py#L772)。

## 运行注意事项

### 1. 模型类别必须匹配

如果模型类别既不包含 `meter_data_9k` 的 5 类，也不包含旧版 `legacy` 的 3 类，脚本会直接报错。

检查逻辑见 [dial_reading.py](/data/prj/yolov8_dial_reading/ultralytics/dial_reading.py#L707)。

### 2. 缺少必要检测时会报错

如果某张图中未检测到必要类别，脚本会报：

`Missing required detections`

对应逻辑见 [dial_reading.py](/data/prj/yolov8_dial_reading/ultralytics/dial_reading.py#L737)。

### 3. 当前环境需要可用的 Python 图像依赖

实际运行 `dial_reading.py` 依赖：

- `numpy`
- `opencv-python`
- `matplotlib`
- `ultralytics`

如果当前解释器里缺少 `numpy`，`cv2` 将无法导入，脚本不能正常推理。

## 适用场景

这个脚本适合：

- 单张图像调试
- `meter_data_9k` 新标注逻辑验证
- 模型效果快速验证
- 批量验证集结果导出

如果后续要把 `pointer_tip` 从“框中心点”切换成“基于直线检测的细化点”，建议在当前脚本上只调整 `meter_data_9k` 模式下的指针点计算逻辑，不影响其余流程。
