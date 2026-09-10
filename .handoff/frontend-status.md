# Frontend status
updated: 2026-09-10T16:15
milestone: FE-M7（Krea 2 模型族接入，数据驱动）
status: done

## Done
- ✅ **类型再生成**：`generated.ts` 用新 openapi.json（66 schemas，含 FamilyInfo / FamilyPreset）重新生成；`types.ts` 新增 `FamilyInfo` / `FamilyPreset` 别名；`train-schema.json` 静态副本已用活后端 `/api/schema/train` 刷新（family 枚举含 krea2、objective 含 res_shift_*）。
- ✅ **A. 族数据源统一**：新增 `src/api/hooks/useFamilies.ts`（TanStack Query 缓存 `/api/families`）；Models.tsx 的写死 `FAMILIES` 常量已移除，族名展示用 `label`。
- ✅ **B. 训练配置页（SchemaForm 新增 `family` prop）**：
  1. `adapter.preset` → 下拉：选项 = 当前族 `presets`，显示 `name — description（N 层）`（data-testid `adapter-preset-select`）。
  2. `dataset.text_encoding` → 选项 = 族 `text_modes`；krea2 下无 online 并显示"仅支持 自动/预缓存"提示（`text-encoding-select`）。
  3. TrainConfig 副作用：切族后 `adapter.preset` 不在新族列表时自动回退 `default_preset`；`text_encoding` 不在 `text_modes` 时自动回退 `auto`（避免后端 400）；应用 `krea2-lokr-default` 等预设时合法值（`preset: all-linear`）不被覆盖。
  4. `model.dit_path / text_encoder_path / vae_path`：标题用 `weights[].label`，下方显示 `weights[].hint` 中文说明（`weight-hint-*`）。
  5. `sampling.steps/cfg/shift` 占位取自族默认；`sampling.shift=null`（krea2）显示"自动（按分辨率）"（含 anyOf 分支）。
- ✅ **C. 模型页**：族下拉与 kind 选项从 `/api/families` 来（`weights[].field` 映射 kind）；`dtype=fp8` 显示紫色 fp8 标签（`fp8-tag`）。
- ✅ **D. i18n**：zh/en 两份补齐 `family.*`、`textMode.*`（含 autoOnly）、`sampling.shiftAuto`、`preset.layers`；i18n 模块对无 localStorage 环境做了防御（测试环境初始化安全）。
- ✅ **E. MSW**：`/api/families`（anima/krea2/toy 真实形状）+ `/api/families/:name`；模型 mock 增加 `family:'krea2', dtype:'fp8'`；预设 mock 增加 `krea2-lokr-default` / `krea2-lora-32`。
- ✅ **F. 验收**：新增 `tests/families.test.tsx` 6 个用例（preset 下拉随族切换、krea2 无 online、anima 含 online、shift 自动占位、weights hint 渲染、krea2-lokr-default 预设应用后 adapter.preset=all-linear）。真实后端截图 `25-krea2-train-config.png`（krea2 预设选中、weights hint 中文提示、Plan 面板真实估算 189.94 M / 15 GB）、`26-krea2-models.png`（Krea 2 Raw 12.9B 行 + fp8 标签 + is_default 星标）。

## In progress / Not done
- 无。objective.res_shift_tokens / res_shift_mu 依赖 schema 的 x-ui（show_when）渲染，已由通用引擎处理，未做额外 UI。

## How to run
- 后端：`cd xiangmuyuanma && ./studio.sh dev`（或 `venv/bin/ypuddin serve --port 8765 --data-root /tmp/ypuddin_data`）
- 前端真实后端：`cd frontend && VITE_USE_MOCK=false npm run dev`；MSW：`npm run dev`

## Tests
- `npm run lint`: pass (0 warnings)
- `npm run test`: 52 passed (9 files)
- `npm run build`: pass

## Questions / blockers for backend
- 无阻塞。观察项：`POST /api/models` 返回的 `exists:false`（路径不存在）符合预期；若希望注册时强制校验路径存在性，请告知策略（当前 UI 用 missing 徽标提示即可）。
