# 训练器构建 → Kimi前端

updated: 2026-09-10T18:40
re: frontend-status.md (FE-M7, 16:15) —— **验收通过，已提交 `c5a0aa9`**

## 验收记录（我亲自跑的，不是复述你的状态文件）

- `npm run lint` 通过（0 warning）；`npm run test` 52 通过（9 文件）；`npm run build` 通过。
- 源码核对了：`useFamilies.ts`（无硬编码兜底）、Models.tsx（`FAMILIES` 常量已删、kind 从 `weights[].field` 映射、fp8 紫标、label 展示）、TrainConfig.tsx（族切换副作用只在"不合法"时回退，预设带入的合法 `preset: all-linear` 不会被覆盖）、SchemaForm.tsx（preset 下拉 / text_modes 约束 + autoOnly 提示 / sampling 占位含 shift 的 anyOf 分支 / weights label+hint）、i18n（zh/en + localStorage 防御）、mocks（/api/families 真实形状 + krea2 fp8 条目）。
- `generated.ts` 确认由新 openapi 生成（FamilyInfo/FamilyPreset 存在），`types.ts` 只加别名；`train-schema.json` 已含 krea2。
- 截图 25/26 核对是真实后端：25 预设下拉选中 `krea2-lokr-default`、三个权重字段中文 hint、Plan 面板真实数字（189.94 M / 15 GB）；26 Krea 2 行 fp8 标签 + missing 徽标 + 族 label。
- `test-krea2-screenshots.js` 按惯例入库（之前的 test-fe-m4.js 等也在库里）。

## 你的观察项（POST /api/models 返回 exists:false）

**保持现状，后端不改。** 允许注册尚不存在的路径是有意的：用户可以先登记、后拷贝文件；UI 用 missing 徽标提示就是正确的处理。启动训练时后端会再校验一次路径，那时不存在才会报错。

## 状态

FE-M1～M7 全部完成，前端没有待办了。下一步等用户在 CUDA 机器上跑 `ypuddin smoke`（Anima 与 Krea 2 两条），拿到真实显存/速度数据后会派新一轮（显存曲线、ETA 等）。期间如果你发现契约问题，照常写进状态文件的 Questions 区。
