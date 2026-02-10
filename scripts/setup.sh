#!/bin/bash
# 一人公司开发环境快速配置脚本

set -e

echo "=== 一人公司开发环境配置 ==="

# 检查 Node.js
if command -v node &> /dev/null; then
    echo "[OK] Node.js $(node -v)"
else
    echo "[!] 未安装 Node.js，请先安装: https://nodejs.org/"
    exit 1
fi

# 检查 Git
if command -v git &> /dev/null; then
    echo "[OK] Git $(git --version)"
else
    echo "[!] 未安装 Git"
    exit 1
fi

# 检查 GitHub CLI
if command -v gh &> /dev/null; then
    echo "[OK] GitHub CLI $(gh --version | head -1)"
    if gh auth status &> /dev/null; then
        echo "[OK] GitHub 已登录"
    else
        echo "[!] GitHub 未登录，请运行: gh auth login"
    fi
else
    echo "[!] 未安装 GitHub CLI，请先安装: https://cli.github.com/"
fi

# 检查 Claude Code
if command -v claude &> /dev/null; then
    echo "[OK] Claude Code 已安装"
else
    echo "[!] 未安装 Claude Code，安装命令: npm install -g @anthropic-ai/claude-code"
fi

echo ""
echo "=== 配置完成 ==="
