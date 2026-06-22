# HYB 涡轮盘截面拓扑结果 - NURBS 参数化重构代码架构说明

## 1. 当前进度与架构边界

当前已经具备两个关键输入：

- `export1.txt`：节点编号到拓扑密度的映射，当前共有 1050 条密度记录。
- `NLIST.lis`：ANSYS 节点坐标列表，当前共有 1050 个节点。

两份文件的节点编号范围均为 1-1050，能够一一合并。坐标当前以 m 为单位输出，代码内部建议统一转换为 mm。当前坐标范围约为：

- X：-17.0 mm 到 17.0 mm
- Z：-8.0 mm 到 11.0 mm
- Y：-1.0 mm 到 0.1 mm，存在多个 Y 层

论文中的几何重构对象是二维轴对称截面，因此第一阶段代码应以二维截面重构为主。当前最稳妥的默认截面投影为 X-Z 平面，Y 方向作为厚度/层方向处理，可以选择 `slice` 模式或 `aggregate` 模式。对于当前文件，建议先使用 `aggregate` 模式，将相同 X-Z 投影点的密度按 `max` 或 `mean` 聚合；若后续明确某一 Y 层才是目标截面，则切换为 `slice`。

本架构覆盖：

```text
ANSYS 节点/密度导入
-> 坐标与密度合并
-> 二维截面投影
-> 密度阈值分割
-> 边界等值线提取
-> 边界清理、排序、重采样
-> NURBS/B-spline 拟合
-> 几何误差评价
-> 曲线/图像/JSON/CAD/ANSYS 再分析接口导出
```

不在第一阶段强制完成但预留接口：

```text
二维截面绕轴生成三维盘体
-> STEP/IGES/STL/APDL 导出
-> ANSYS 再分析
-> 多工况样本批处理
-> 神经网络代理模型数据集生成
```

## 2. 论文约束转化为代码需求

论文第 4 章的路线不是“导出全部节点”，而是“密度阈值分割 - 边界轮廓提取 - 点集排序 - 弧长重采样 - NURBS 拟合”。因此代码中不能直接把所有高密度节点当作曲线点拟合，否则会把内部材料点也混进边界，得到不可控的曲线。

代码中的核心数学对象应对应论文中的两个集合：

```text
保留材料区域: Ω_eta = {x in Ω | rho(x) >= eta}
边界轮廓:     Γ_eta = {x in Ω | rho(x) = eta}
```

默认阈值 `eta = 0.50`，因为论文参数表中给出的密度阈值为 0.50。当前 `export1.txt` 在该阈值下约有 699 个节点被判为保留材料。这个数字只用于 sanity check，不能直接作为边界点数。

NURBS 拟合建议从论文中的保守配置开始：

```text
degree = 3
weights = 1.0  # 第一阶段退化为 B-spline，但数据结构保留 NURBS 权重
knot vector = clamped/open uniform 或 closed periodic
error target:
  mean error <= 0.5 mm
  max error  <= 1.5 mm
  area error <= 3%
```

## 3. 总体模块结构

推荐包结构如下：

```text
hyb_nurbs/
  cli.py
  config.py
  schema.py
  pipeline.py

  io/
    ansys.py

  preprocess/
    projection.py

  boundary/
    iso.py
    postprocess.py

  nurbs/
    fitting.py
    evaluate.py

  validation/
    metrics.py

  cad/
    revolve.py

  exporters/
    files.py

  viz/
    plots.py

configs/
  default.yaml

tests/
  test_io.py
  test_geometry.py
```

模块职责如下：

```text
io.ansys
  解析 NLIST.lis 与 export1.txt，按 node_id 合并，完成单位转换和基础校验。

preprocess.projection
  将 3D 节点云投影到二维截面。默认 axes=(x,z)，Y 作为层方向。
  支持 aggregate 和 slice 两种模式。

boundary.iso
  从二维散点密度场提取 rho=eta 的边界等值线。
  当前无单元连接信息时默认用 Delaunay 三角剖分等值线或 griddata + marching squares。
  后续若导出 ELIST/CDB，则切换为 mesh-aware marching triangles/quads。

boundary.postprocess
  闭合轮廓、过滤小连通域、过滤小孔洞、统一方向、按弧长重采样、必要时按角点切段。

nurbs.fitting
  对每条边界 loop 或每个 segment 做三次 NURBS/B-spline 最小二乘拟合。
  控制点数自适应增加，直到误差达到阈值或到达最大控制点数。

nurbs.evaluate
  提供 Cox-de Boor 基函数、NURBS 曲线采样、导数/曲率接口。

validation.metrics
  计算 mean/max/Hausdorff 距离误差、面积偏差、闭合误差、自交检查。

cad.revolve
  将二维 NURBS 截面绕轴生成三维表面/实体或采样网格。

exporters.files
  输出 CSV、JSON、APDL 宏、CAD 中间文件。

viz.plots
  输出密度云图、阈值边界叠加图、NURBS 拟合误差图。
```

## 4. 核心数据结构

建议所有模块通过 dataclass 传参，避免到处传裸数组。

```python
@dataclass
class NodeDensityTable:
    node_id: np.ndarray   # [N]
    xyz: np.ndarray       # [N, 3], mm
    rho: np.ndarray       # [N]

@dataclass
class SectionCloud:
    point_id: np.ndarray
    xy: np.ndarray        # [M, 2], usually X-Z in mm
    rho: np.ndarray
    source_node_ids: list[list[int]]
    axes: tuple[str, str] = ("x", "z")

@dataclass
class BoundaryLoop:
    points: np.ndarray    # [K, 2]
    role: str             # outer | hole | unknown
    component_id: int
    is_closed: bool = True

@dataclass
class NurbsCurveSpec:
    degree: int
    control_points: np.ndarray  # [C, 2]
    weights: np.ndarray         # [C]
    knots: np.ndarray           # [C + degree + 1]
    is_closed: bool
    role: str
    segment_id: int

@dataclass
class FitMetrics:
    mean_error_mm: float
    max_error_mm: float
    hausdorff_error_mm: float
    area_error_ratio: float
    n_control_points: int
    n_samples: int
```

## 5. 算法主流程

### 5.1 文件解析

输入文件解析必须满足以下规则：

```text
NLIST.lis:
  忽略 ANSYS 的重复表头与空行。
  只接受形如 node_id x y z 的数据行。
  node_id 必须唯一。

export1.txt:
  忽略表头。
  只接受形如 node_id rho 的数据行。
  node_id 必须唯一。

合并:
  使用 inner merge 并检查合并后行数。
  如果节点文件和密度文件不完全匹配，直接抛错，不允许静默丢点。
```

当前文件的解析 sanity check：

```text
n_nodes = 1050
n_density = 1050
node_id_min = 1
node_id_max = 1050
```

### 5.2 单位与投影

坐标由 ANSYS 输出为 m。由于论文中的几何误差指标使用 mm，代码内部统一转成 mm。

投影策略：

```text
默认:
  section_axes = (x, z)
  thickness_axis = y
  projection_mode = aggregate
  aggregate_density = max

备选:
  projection_mode = slice
  slice_y = 用户指定的目标 Y 层
```

`aggregate_density=max` 的含义是：如果多个 Y 层投影到同一个 X-Z 点，只要任一层密度高，就保留该点的材料趋势。它适合初期几何提取。若后续明确设计对象是某个严格截面，则改为 `slice`。

### 5.3 密度场等值线提取

当前只有节点坐标与节点密度，没有单元连接关系。因此第一阶段推荐两套可切换算法。

方案 A：Delaunay 三角剖分等值线，默认优先。

```text
输入 SectionCloud(xy, rho)
1. 用 scipy.spatial.Delaunay 对 xy 建三角剖分。
2. 对每个三角形检查 rho 是否跨越 eta。
3. 若跨越，则在线性插值的三角形边上求交点。
4. 每个三角形最多产生一条等值线线段。
5. 将所有线段按端点距离 stitch 成 polyline/loop。
6. 删除无法闭合或点数过少的碎片。
```

方案 B：规则网格插值 + marching squares，作为 fallback。

```text
1. 对 section bbox 建规则网格，网格间距默认 0.10-0.20 mm。
2. 用 scipy.interpolate.griddata 或 RBF 将 rho 插值到网格。
3. 用 skimage.measure.find_contours 提取 rho=eta 的等值线。
4. 将图像坐标转换回 X-Z 坐标。
```

方案 C：alpha shape，仅作为临时备选。

```text
1. 选取 rho >= eta 的点。
2. 计算 concave hull。
3. 得到材料区域外边界。
```

C 不能很好表达灰度过渡区的 rho=eta 边界，也不适合复杂孔洞，只能作为 debug 或没有 scipy/skimage 时的后备。

### 5.4 边界后处理

边界输出进入 NURBS 前必须经过以下清理：

```text
1. close loop
   若首尾距离 <= close_tolerance，首尾合并；否则丢弃或尝试补线。

2. filter by area
   面积小于 min_component_area_mm2 的连通域删除。
   面积小于 min_hole_area_mm2 的孔洞填补或忽略。

3. orientation
   外边界统一为 CCW，孔洞统一为 CW，便于后续 CAD/布尔操作。

4. self-intersection check
   自交 loop 不允许直接拟合，必须先修复或分裂。

5. arclength resampling
   默认间距 0.25 mm。NURBS 拟合不直接使用原始非均匀点列。

6. corner split
   若存在真实几何角点，按夹角阈值切段；角点处用 C0 拼接，平滑段内部保持 C1/C2。
```

### 5.5 NURBS/B-spline 拟合

第一阶段推荐固定权重为 1，即先做 B-spline 拟合，但输出格式仍然叫 `NurbsCurveSpec`，以便第二阶段加入权重优化。

拟合流程：

```text
for each BoundaryLoop:
  points = arclength_resample(loop)
  if corner_split_enabled:
      segments = split_by_curvature_or_corners(points)
  else:
      segments = [points]

  for each segment:
      n_ctrl = max(min_ctrlpts, ceil(segment_length / initial_control_spacing))
      while n_ctrl <= max_ctrlpts:
          u = chord_length_parameterization(points)
          knots = open_uniform_knot_vector(n_ctrl, degree)
          N = basis_matrix(u, degree, knots)
          solve min ||N P - Q||^2 + lambda_smooth ||D2 P||^2
          evaluate error
          if error <= tolerance:
              break
          n_ctrl += refine_step
```

误差评价：

```text
mean_error = mean nearest distance from source boundary points to fitted samples
max_error = max nearest distance
hausdorff_error = max(directed source->fit, directed fit->source)
area_error_ratio = |A_fit - A_topo| / A_topo
```

停止准则建议：

```text
mean_error_mm <= 0.50
max_error_mm <= 1.50
area_error_ratio <= 0.03
```

### 5.6 输出

每次运行应生成独立输出目录，例如 `outputs/run_001/`。

推荐输出：

```text
merged_node_density.csv
section_cloud.csv
boundary_loops.json
nurbs_fit_results.json
fit_metrics.csv

density_cloud.png
boundary_overlay.png
nurbs_fit_overlay.png
fit_error_heatmap.png

optional:
  section_curves.dxf
  section_curves.iges
  section_curves.step
  sampled_section_for_apdl.mac
  revolved_surface.stl
  revolved_solid.step
```

JSON 中的 NURBS 曲线至少包含：

```json
{
  "degree": 3,
  "control_points": [[x0, z0], [x1, z1]],
  "weights": [1.0, 1.0],
  "knots": [0, 0, 0, 0, 0.5, 1, 1, 1, 1],
  "is_closed": true,
  "role": "outer",
  "segment_id": 0
}
```

## 6. 推荐配置

第一轮可使用下面的配置。

```yaml
input:
  node_file: NLIST.lis
  density_file: export1.txt
  density_is_node_based: true

units:
  input_length_unit: m
  working_length_unit: mm
  auto_scale_to_mm: true

projection:
  axes: [x, z]
  thickness_axis: y
  mode: aggregate
  slice_y: null
  aggregate_density: max
  aggregate_coord: mean

threshold:
  eta: 0.50
  adaptive: false
  min_component_area_mm2: 1.0
  min_hole_area_mm2: 0.5

boundary:
  method: tri_iso
  grid_resolution_mm: 0.15
  smoothing_window: 5
  resample_spacing_mm: 0.25
  close_tolerance_mm: 0.10
  curvature_split_enabled: true
  corner_angle_deg: 135

nurbs:
  degree: 3
  closed_loop_mode: periodic
  initial_control_spacing_mm: 2.0
  min_ctrlpts: 6
  max_ctrlpts: 80
  weights_mode: fixed_ones
  lambda_smooth: 1.0e-4
  fit_tolerance:
    mean_mm: 0.50
    max_mm: 1.50
    area_ratio: 0.03
  max_refine_iters: 8

export:
  out_dir: outputs/run_001
  write_debug_csv: true
  write_boundary_json: true
  write_nurbs_json: true
  write_plots: true
  write_apdl_macro: false
  write_cad: false
```

## 7. Codex 实施顺序

请 Codex 按以下顺序实现，避免一开始就碰 CAD 导出。

```text
Step 1: 实现 io.ansys
  - parse_nlist
  - parse_density
  - load_node_density
  - 用当前 NLIST.lis/export1.txt 做测试，必须解析出 1050 条并完全匹配。

Step 2: 实现 preprocess.projection
  - aggregate 模式
  - slice 模式
  - 输出 section_cloud.csv

Step 3: 实现 boundary.iso 的 tri_iso
  - Delaunay 三角剖分
  - eta 交线插值
  - 线段 stitching
  - loop closure

Step 4: 实现 boundary.postprocess
  - 面积过滤
  - loop 方向
  - 弧长重采样
  - 自交检查

Step 5: 实现 viz.plots
  - density_cloud.png
  - boundary_overlay.png
  先用图像确认边界正确，再进入 NURBS。

Step 6: 实现 nurbs.evaluate 与 nurbs.fitting
  - B-spline basis matrix
  - open uniform knot vector
  - least-squares control point solve
  - adaptive control-point refinement

Step 7: 实现 validation.metrics
  - mean/max/Hausdorff/area error
  - 输出 fit_metrics.csv

Step 8: 实现 exporters.files
  - JSON 序列化
  - sampled APDL macro 作为 CAD 不可用时的备用路线

Step 9: 实现 cad.revolve
  - 先做采样网格 STL
  - 再做 STEP/IGES 或 pythonocc/geomdl 路线
```

## 8. 工程注意事项

1. 不要把 `rho >= eta` 的所有节点直接传给 NURBS。高密度节点包括材料内部点，不是边界点。

2. 当前没有单元连接关系，等值线算法要基于散点三角剖分或网格插值。若后续能够导出 `ELIST.lis`、`.cdb` 或元素-节点连接表，必须优先使用 mesh-aware contour，因为它比散点插值更接近有限元网格拓扑。

3. 坐标单位必须统一。当前 ANSYS 文件是 m，论文误差指标是 mm。所有拟合和误差计算在 mm 中完成。

4. 对称轴、中心孔、轮缘保留区等工程边界不应盲目平滑。若这些边界是非设计区或解析几何，应通过 mask/feature-boundary 单独保留，NURBS 只拟合自由边界。

5. 第一阶段权重固定为 1。不要过早优化 NURBS 权重，否则容易把算法调试问题和非线性优化问题混在一起。

6. 每一步都要输出 debug 图。当前阶段最重要的是确认 `rho=eta` 边界是否与拓扑云图一致，而不是只看拟合曲线是否平滑。

7. 每个 loop 必须有质量门槛：闭合误差、面积、方向、自交、采样点数、拟合误差。任何一个失败都应进入 report，而不是静默输出 CAD。

## 9. 最小可验收标准

当前阶段代码完成后，至少应满足：

```text
1. 能解析当前两个结果文件，并确认 1050 个节点和 1050 个密度一一对应。
2. 能生成 X-Z 截面密度云图。
3. 能在 eta=0.5 下生成一组闭合边界 loop。
4. 能输出边界点序列 JSON/CSV。
5. 能完成 cubic NURBS/B-spline 拟合。
6. 能输出平均误差、最大误差、Hausdorff 误差、面积偏差。
7. 能输出 NURBS 控制点、权重、节点向量。
8. 能输出拟合叠加图，肉眼检查边界与拟合曲线一致。
```

满足以上标准后，再进入三维旋转建模与 ANSYS 再分析接口。
