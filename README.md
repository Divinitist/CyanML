# CyanML

CyanML 是一个本机运行的机器学习手撕题库，包含 43 道题，覆盖传统机器学习、神经网络、Transformer、LLM 推理及强化学习损失。

界面支持自动保存、标准答案切换、100 组测试、逐测试点输入输出、最近得分缓存，以及可拖动的三栏布局。

## 环境要求

- Windows 10/11
- Python 3.12
- Node.js 18+，建议使用当前 LTS 版本
- npm
- uv

从 [Node.js 官网](https://nodejs.org/en/download)安装 LTS 版本。然后在 PowerShell 中安装 uv：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

重新打开终端并确认：

```powershell
node --version
npm --version
uv --version
```

## 首次安装

进入仓库并安装前后端依赖：

```powershell
cd D:\Projects\Sion0\ML
uv sync --python 3.12
cd frontend
npm ci
cd ..
```

Python 依赖由 `uv.lock` 固定，前端依赖由 `package-lock.json` 固定。

## 启动与停止

在仓库根目录使用统一脚本：

```bat
cyanml.bat start
cyanml.bat status
cyanml.bat restart
cyanml.bat stop
```

启动后访问：

- 前端：<http://127.0.0.1:5173>
- API：<http://127.0.0.1:8765>
- API 文档：<http://127.0.0.1:8765/docs>

首次执行 `start` 时，如果缺少 `.venv` 或 `node_modules`，脚本也会自动安装相应依赖。

## 使用方法

1. 在左侧选择题目并阅读题面。
2. 在“我的作答”中实现指定函数，代码会自动保存。
3. 点击“运行 100 组测试”，查看每个测试点的输入、实际输出、预期输出和错误信息。
4. 点击“查看标答”阅读带逐行解释的标准实现。
5. “重置”会使用该题的 `template.py` 覆盖当前作答。

题目默认使用 `float64`，数值误差由每题单独设置；随机题通过显式随机种子保证可复现。

## 测试与构建

运行全部 Python 测试：

```powershell
.\.venv\Scripts\python.exe -m pytest
```

构建前端：

```powershell
cd frontend
npm run build
```

当前没有前端单元测试文件。如需检查 Vitest 配置：

```powershell
cd frontend
npm test -- --passWithNoTests
```

## 目录结构

```text
ML/
├─ backend/cyanml_api/   FastAPI 后端与评测器
├─ frontend/src/         React 前端
├─ questions/            题面、模板、作答、标答和测试
├─ tests/                Python 回归测试
├─ scripts/cyanml.ps1    进程管理脚本
└─ cyanml.bat            统一启动入口
```

每道题包含：

```text
manifest.json   元数据与展示顺序
problem.md      题面
template.py     初始模板
submission.py   当前作答
answer.py       标准答案
test.py         100 组测试
```

## 本地数据

- 作答：`questions/<题目>/submission.py`
- 最近得分：`questions/.scores.json`
- 最近评测：`questions/.results/<id>.json`
- 运行日志：`.cyanml/*.log`
- 未同步草稿和面板尺寸：浏览器 `localStorage`

## 安全提示

项目没有登录、鉴权或网络级代码沙箱，只应在本机使用。不要把 5173/8765 端口暴露到局域网或公网，也不要运行不可信代码。
