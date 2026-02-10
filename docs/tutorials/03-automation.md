# 03 - 自动化工作流

## Hooks：生命周期自动化

在 `.claude/settings.json` 中配置：

```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Edit|Write",
      "hooks": [{
        "type": "command",
        "command": "npx prettier --write \"$CLAUDE_FILE_PATH\""
      }]
    }],
    "Notification": [{
      "matcher": "",
      "hooks": [{
        "type": "command",
        "command": "notify-send 'Claude Code' '需要你的注意'"
      }]
    }]
  }
}
```

## 自定义 Skills：固化常用流程

创建 `.claude/skills/deploy/SKILL.md`：

```markdown
---
name: deploy
description: 部署应用到生产环境
allowed-tools: Bash
---
执行部署流程：
1. 运行测试确保通过
2. 构建生产版本
3. 推送到服务器
4. 验证部署状态
```

## 无头模式：脚本化 Claude

```bash
# 自动代码审查
git diff | claude -p "审查这些代码变更，指出潜在问题"

# 自动生成 commit message
claude -p "查看暂存的变更并创建 commit" \
  --allowedTools "Bash(git *)"

# 批量处理日志
cat error.log | claude -p "分析错误模式并给出修复建议" \
  --output-format json
```

## GitHub Actions 集成

```yaml
name: Claude Code Review
on:
  pull_request:
    types: [opened, synchronize]
jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
```
