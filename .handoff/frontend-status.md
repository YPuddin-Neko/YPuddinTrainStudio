# Frontend status
updated: 2026-09-10T10:25
milestone: FE-M4（数据集 + 项目详情 + 模型 + 设置）
status: done

## Done
- ✅ **契约对齐**：时间字段统一按 Unix 秒处理；新增 `job.sample_progress` / `job.event` / `artifact.created` 事件类型；`Plan` 类型按真实形状更新（buckets.items、顶层 images/items/captioned、gpu_total_mb 可空）。
- ✅ **A. 数据集页 `/datasets/:id`**（`pages/Dataset/Dataset.tsx`）：
  - A1 概览卡：images/覆盖率/masks/缓存命中、resolutions 与 ar_hist 直方图；`index_status=indexing` 显示进度条（监听 `job.cache_progress` kind=index、按 dataset_id 匹配）；`dataset.changed` 后自动刷新（`13-dataset-overview.png`）。
  - A2 虚拟滚动图片网格（自实现窗口化渲染，不依赖额外库；缩略图 `GET /datasets/{id}/images/{hash}/thumb?size=256`，懒加载 + 滚动近底自动翻页）。
  - A3 caption 编辑器：TagChips（解析/序列化/去重/增删改/HTML5 拖拽排序，逻辑在 `utils/tags.ts` 全部独立可测），保存 `PUT caption`（`14-caption-editor.png`）；多选后批量 `POST tags/batch`（`15-batch-tags.png`）；`q=` 服务端过滤（`16-search-filter.png`）。
  - A4 顶部动作：Rescan / Remove（二次确认，不删磁盘文件）/ Pre-cache（POST /jobs type=cache，使用项目草稿配置）。
  - A5 分桶预览：用 `GET /projects/{pid}/config` 草稿调 `POST /plan` 展示 buckets（w×h、items、batches）。
- ✅ **B. 项目详情 `/projects/:id`**：四个 tabs——数据集（卡片列表 + 注册表单：path 浏览/repeats/caption_ext/is_reg/prior_weight/class_prompt，已适配后端返回 `DatasetInfo[]` 形状取 `.source`）、训练配置（跳转）、任务（`GET /jobs?project_id=`）、产物（`GET /artifacts?project_id=`）（`12`, `18`）。
- ✅ **C. 模型权重页 `/models`**：列表（family/kind/path/size/dtype/exists/is_default）+ 添加（路径浏览 PathBrowser）+ 扫描目录 + 删除；SchemaForm 的 `dit_path/text_encoder_path/vae_path/tokenizer_path` 控件带"从已注册模型选择"下拉（`19-models.png`）。
- ✅ **D. 设置页 `/settings`**：paths/server/ui 完整表单，`PUT /settings` 保存后语言/主题即时生效（`20-settings.png` 英文界面实证）。
- ✅ **JobDetail 采样进度**：订阅 `job.sample_progress`，header 区显示"生成预览 第 k/n 张 · 步 done/total"进度条，预览阶段不再像卡死。
- ✅ **项目配置草稿**：TrainConfig 加载 `GET /projects/{id}/config` 并 1s 防抖 `PUT` 自动保存（预缓存/Plan 与草稿一致）。
- ✅ **MSW handlers 补齐**：datasets（info/images/thumb/caption/tags/batch/rescan/delete）、models（CRUD+scan）、settings PUT、jobs 分页+创建+project_id 过滤、artifacts 过滤/convert。
- ✅ **测试**：`tests/tags.test.ts`（9 例：解析/序列化/去重/拖拽/批量合并）、`tests/useDatasetImages.test.tsx`（5 例：分页/loadMore/q 过滤/多选/caption 更新，MSW 驱动）。

## In progress / Not done
- Queue 拖拽调优先级（目前是数字输入框 onBlur PATCH）、任务详情 validation 曲线图与 throughput 图分离、日志 5 万行环形缓存裁剪（目前是纯追加）。
- Mock SSE 推送生成器（开发态看曲线自增）；FE-M4 要求的真实验证里用的是真后端，未受影响。

## How to run
- 后端：`cd xiangmuyuanma && .venv/bin/ypuddin serve --port 8765 --data-root /tmp/ypuddin_data`
- 前端真实后端：`cd frontend && VITE_USE_MOCK=false npm run dev`；MSW：`npm run dev`
- E2E：`node test-real-flow.js`（FE-M5 训练流）、`node test-fe-m4.js`（FE-M4 数据集/模型/设置流）

## Tests
- `npm run lint`: pass (0 warnings)
- `npm run test`: 37 passed (7 files)
- `npm run build`: pass（各页面独立 chunk + echarts 独立 chunk）
- 真实后端 E2E：`12~20` 共 9 张新截图（注册→索引 ready→caption chips 编辑保存→批量 tag（验证 q=batch_m4_tag 命中 2 张）→预缓存发起→模型添加→设置语言切换英文即时生效）

## Questions / blockers for backend
1. **【Bug 需修】cache 类型任务退出码 0 被标记 failed**：`j_6bfa29ad695c`（type=cache）进程 exit_code=0、run.log 正常输出 bucket 摘要结尾，但 job 被置为 `failed`，error 显示 `process exited with code 0: }`（取到 run.log 最后一行"}"当错误）。现象：该 run 未产生 events.jsonl。推测监督器对 cache 任务的完成判定不依赖 exit code 而是等 `run.*` 事件流，cache 任务没走事件管道导致误判。另 `GET /datasets/{id}` 的 `cache` 字段在缓存完成后仍是 `{}`。
2. `GET /projects/{id}/datasets` 返回的是 `DatasetInfo[]`（含 stats/index_status），与 spec §5 表格里写的 `DatasetSource` 列表不一致——前端已按 `.source` 取值兼容，请同步 spec 或改回。
3. `POST /models/scan` 的请求体是自由 map（openapi 里 `additionalProperties: string`），请文档化其语义（传什么键值？目录路径还是无参扫 settings.models_dir）。
