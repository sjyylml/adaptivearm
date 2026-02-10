# 02 - 用 Claude Code 替代一整个团队

## 角色替代矩阵

| 传统角色 | Claude Code 方案 | 关键命令/功能 |
|----------|-----------------|---------------|
| 前端工程师 | 编写 React/Vue/Next.js | 直接 Edit/Write |
| 后端工程师 | API、数据库、服务端逻辑 | Edit + Bash |
| DevOps | Dockerfile、CI/CD、部署 | Bash + Skills |
| 代码审查 | 自动审查 PR | `/review` |
| QA 测试 | 编写并运行测试 | Bash(pytest/jest) |
| 安全工程师 | 安全审查 | `/security-review` |
| 技术写作 | API 文档、README | Write |

## 实战示例

### 用一句话创建完整 API

```
帮我创建一个用户认证 API，使用 Express + JWT + PostgreSQL，包含注册、登录、刷新 token 三个端点
```

### 用一句话修复 Bug

```
报错信息：TypeError: Cannot read property 'id' of undefined at line 42 of auth.js，帮我排查并修复
```

### 用一句话部署

```
帮我写一个 Dockerfile 和 docker-compose.yml，包含 Node.js 应用和 PostgreSQL 数据库
```
