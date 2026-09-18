# 文件夹布局清单

训练器有两棵独立的目录树：**源码树**（跟着 Git 更新）和**数据根**（你的项目、模型和产物，不随代码更新迁移）。
两棵树里各有一个名叫 `environment` 的目录，用途完全不同 —— 见 [§3](#3-两个-environment-分别是什么)。

本清单里每一条都标注了决定该路径的代码位置，改代码时请一并更新这里。

## 1. 源码树

`<repo>` 指仓库根目录（即 `xiangmuyuanma/`）。

```
<repo>/
├── studio-windows-cuda.bat      Windows + NVIDIA CUDA，x86_64
├── studio-cpu.bat               Windows 仅 CPU，x86_64 / arm64
├── studio-linux-cuda.sh         Linux + NVIDIA CUDA，x86_64
├── studio-linux-dtk.sh          Linux + 海光 DTK，x86_64（直接调用 bootstrap.py）
├── studio-macos.command         macOS Apple 芯片（MPS），arm64
├── studio-cpu.sh                Linux / macOS 仅 CPU，x86_64 / arm64
│                                以上 6 个是唯一的启动入口，依赖互不共用
├── scripts/
│   ├── launch.bat / launch.sh   启动入口的共用阶段：只负责找到可用的 Python，不是入口
│   ├── bootstrap.py             全部部署逻辑（建环境、装依赖、构建前端、起服务）
│   ├── package_source.py        打包与仓库内容校验
│   └── …                        其他开发辅助脚本
├── ypuddin/                     Python 后端包：训练核心、CLI、FastAPI 服务
├── frontend/
│   ├── src/                     前端源码
│   └── dist/                    构建产物，**不进 Git**；服务就是从这里取页面
├── environment/<profile>/venv   各环境的 Python 虚拟环境（见 §2）
├── venv/                        旧版部署的虚拟环境，对应 legacy 环境类型
├── tests/                       后端测试；前端测试在 frontend/tests/
├── docs/                        部署、设计与验证文档
└── studio_data/                 默认数据根（见 §4），可用 --data-root 换到别处
```

**`frontend/dist/` 不在 Git 里**（`.gitignore`）。所以 `git pull` 只会带来前端源码，页面不会变；重启启动入口时 `bootstrap.py` 会比对 `frontend/src` 与配置文件的哈希（`frontend_stale()`），发现过期就跑 `npm ci` + `npm run build`。构建需要 Node 20.19+ 或 22.12+；没装 Node 且 dist 与源码不一致时会直接报错，不会静默使用旧页面。

服务挂载 `frontend/dist` 的位置在 `ypuddin/server/app.py` 的 `_mount_spa()`。

## 2. 部署环境（源码树内）

```
<repo>/environment/<profile>/venv/
└── .ypuddin-install.json        安装标记：环境类型、CPU 架构、torch 版本、依赖签名
```

`<profile>` 取值：`windows-cuda`、`linux-cuda`、`linux-dtk`、`macos-mps`、`windows-cpu`、`linux-cpu`、`macos-cpu`。
旧版部署是 `legacy`，环境在 `<repo>/venv/`。

- 目录选择：`scripts/bootstrap.py` 的 `select_environment()`。
- 安装标记同时记录**环境类型和 CPU 架构**；换环境类型或换架构再启动会直接拒绝并说明原因，不会覆盖别人的环境。
- 三个加速入口（CUDA、DTK）只在 x86_64 上验证过，在 arm64 上启动会拒绝并提示改用 CPU 入口；MPS 入口要求 Apple Silicon。

## 3. 两个 `environment` 分别是什么

| 路径 | 谁创建 | 装什么 |
| --- | --- | --- |
| `<repo>/environment/` | 启动入口（`bootstrap.py`） | 部署用的 Python 虚拟环境。属于**代码侧**，跟着仓库走 |
| `<数据根>/environment/` | 运行中的服务 | 界面里「运行环境」管理出来的**额外 PyTorch 运行时**和服务控制文件。属于**数据侧** |

数据根里那个的结构（`ypuddin/runtime_profiles.py` 的 `profile_root()`）：

```
<数据根>/environment/
└── <profile>/                   legacy 环境没有这一层，内容直接放在 environment/ 下
    ├── runtimes/<id>/           从界面安装的备选 PyTorch 运行时（ypuddin/server/torch_environments.py）
    ├── service/
    │   ├── selected.json        服务下次用哪个运行时启动
    │   └── restart-*.json       重启请求（ypuddin/server/lifecycle.py）
    ├── cache/                   安装器下载缓存
    └── installer                安装器工作文件（ypuddin/server/environment.py）
```

`runtimes/<id>/` 本身也是虚拟环境（`torch_environments.py` 用 `python -m venv` 创建），不是缓存。
它和源码侧那个的区别在归属：源码侧的基础环境由启动入口按代码重建，数据根这些是运行期产物，
可以跟着 `--data-root` 放到大盘上，并在源码树被整体替换后继续存在。

已知不彻底的地方：基础环境不受 `--data-root` 控制，体积最大的那个环境只能待在源码树旁边。
同类问题见 [存储布局审计](STORAGE_LAYOUT_AUDIT_2026-09-13.md)（改 `cache_dir` 不迁移 `environment/cache`）。

两棵树都不再有 `profiles/` 这一层。旧版本装在 `environment/profiles/<profile>/` 下；
启动入口发现这个旧目录时会打印它的路径并照常安装到新位置，**不会自动删除**，确认新环境可用后自行删除即可。

## 4. 数据根

默认 `<repo>/studio_data/`，用 `--data-root` 可换。代码更新不会迁移这里的任何内容。

```
<数据根>/
├── studio.db                    SQLite 主库：项目、版本、任务、键值表
├── settings.json                服务设置（路径、地址、界面偏好）
├── secrets.json                 模型站点凭据（ypuddin/server/model_credentials.py）
├── project/<项目 id>/           当前布局（layout_version ≥ 2）
│   └── v<N>/                    版本目录，按版本号命名
│       ├── config.json          该版本的训练配置
│       ├── traindata/           训练图片
│       ├── reg/                 正则图片
│       ├── cache/               该版本的潜变量 / 文本缓存
│       ├── samples/             训练中的采样预览
│       └── output/              该版本的训练产物
├── projects/<项目 id>/          旧布局（layout_version < 2）
│   └── versions/<版本 id>/      旧版本目录，另含 datasets/ 与 runs/
├── cache/
│   ├── shared/                  未绑定项目的缓存
│   ├── index.sqlite             图片索引库
│   └── xyz-fingerprints/        模型测试指纹缓存
├── models/                      默认模型目录（设置里可改）
├── runs/                        默认产物目录（设置里可改）
├── presets/                     参数预设
├── datasets/<id>.json           数据集描述文件
├── thumbs/<hash>_<尺寸>.jpg     缩略图缓存
├── environment/                 见 §3
└── network/secrets.json         网络相关凭据（ypuddin/server/network.py）
```

`models/` 下载过程中会出现 `.downloads/<任务 id>/` 暂存目录，完成或取消后清理，只清理该任务自己的暂存文件。

## 5. 升级与备份边界

- **跟着 Git 更新**：源码树除 `environment/`、`venv/`、`frontend/dist/`、`studio_data/` 之外的内容。
- **不随更新迁移**：数据根全部内容，以及已建好的部署环境。
- **备份优先级**：`studio.db`、`settings.json`、`project/`（或旧布局 `projects/`）、`presets/`、`models/`。
- **可安全删除并自动重建**：`frontend/dist/`、`cache/`、`thumbs/`。删除 `environment/` 或 `venv/` 会导致下次启动重装环境。
