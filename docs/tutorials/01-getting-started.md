# 01 - 快速开始：配置你的 AI 开发环境

## 安装 Claude Code

```bash
npm install -g @anthropic-ai/claude-code
```

## 初始化项目

```bash
mkdir my-project && cd my-project
claude
# 进入后运行 /init 生成 CLAUDE.md
```

## 核心概念

### CLAUDE.md — 项目记忆

每个项目根目录的 `CLAUDE.md` 是 Claude Code 的"记忆文件"，它会在每次会话开始时被读取。在这里记录：
- 项目架构决策
- 常用构建/测试命令
- 编码约定

### 斜杠命令

| 命令 | 用途 |
|------|------|
| `/init` | 初始化项目 |
| `/plan` | 进入规划模式（只读分析） |
| `/compact` | 压缩上下文 |
| `/review` | 审查 PR |
| `/memory` | 编辑记忆文件 |

### 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+B` | 后台运行当前任务 |
| `Alt+T` | 开启深度思考 |
| `Shift+Tab` | 切换权限模式 |
| `@` | 文件路径自动补全 |
