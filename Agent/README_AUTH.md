# 公文写作助手 - 使用说明

## 🚀 快速开始

### 1. 启动服务

```bash
cd Agent
export ADMIN_DEFAULT_PASSWORD='<设置强随机密码>'
python app.py
```

服务启动后会自动创建：
- 默认管理员账号
- SQLite 数据库（`./data/agent_memory.db`）

### 2. 访问系统

打开浏览器访问：http://localhost:5003

---

## 👤 账号信息

### 默认管理员账号

| 项目 | 值 |
|------|-----|
| **用户名** | `admin` |
| **密码** | 环境变量 `ADMIN_DEFAULT_PASSWORD` 的值 |
| **角色** | 系统管理员 |

⚠️ **不要在代码或文档中提交真实密码，首次登录后请立即修改！**

---

## 📁 功能说明

### 登录页面 (`/`)
- 用户名/密码登录
- JWT Token 认证
- 首次登录密码修改提示

### 聊天页面 (`/chat`)
- 公文智能生成
- 会话自动保存
- 历史记录恢复
- 用户偏好自动应用

### 管理后台 (`/admin`)
- 用户管理（添加/禁用/重置密码）
- 查看登录日志
- 系统统计

---

## 👥 用户管理

### 添加新用户（管理员）

1. 登录管理员账号
2. 进入管理后台 `/admin`
3. 点击"添加用户"
4. 填写信息：
   - 用户名（登录用，唯一）
   - 密码（初始密码）
   - 姓名（显示用）
   - 部门
   - 角色（user/admin）

### 用户角色权限

| 角色 | 权限 |
|------|------|
| **user** | 使用公文生成、查看自己的历史 |
| **admin** | 全部权限 + 用户管理 + 查看日志 |

---

## 💾 数据存储

所有数据保存在 SQLite 数据库中：

```
./data/
├── agent_memory.db    # 主数据库（用户、会话、消息）
└── faiss_index/       # 知识库向量索引
```

**备份**：直接复制 `agent_memory.db` 文件即可。

---

## 🔧 系统特性

### 安全特性
- ✅ 密码哈希存储（SHA256+salt）
- ✅ JWT Token 认证
- ✅ 会话隔离（严格用户隔离）
- ✅ 管理员权限校验
- ✅ 登录日志记录

### 记忆特性
- ✅ 用户画像（偏好字体、常用公文类型）
- ✅ 会话历史（自动保存对话）
- ✅ 上下文恢复（继续之前的对话）
- ✅ 跨会话搜索（搜索所有历史）

### 团队特性
- ✅ 20人并发支持
- ✅ 部门隔离
- ✅ 会话持久化（SQLite）
- ✅ 数据备份简单

---

## 📊 数据库结构

### 用户表 (auth_users)
- user_id, username, password_hash
- name, department, role
- created_at, last_login

### 会话表 (sessions)
- session_id, user_id
- title, doc_type
- created_at, updated_at
- is_active

### 消息表 (messages)
- id, session_id
- role, content, timestamp
- metadata

---

## 🛠️ 常见问题

### Q: 忘记管理员密码怎么办？

**A**: 删除数据库重新初始化：
```bash
rm ./data/agent_memory.db
python app.py  # 会自动创建新的 admin 账号
```

### Q: 如何修改用户部门？

**A**: 当前版本需要管理员在管理后台删除旧账号，重新创建新账号。

### Q: 会话数据会保存多久？

**A**: 默认永久保存。可以定期运行清理：
```python
memory.cleanup_expired_sessions(days=30)  # 清理30天未活跃的会话
```

### Q: 如何导出用户数据？

**A**: 在 Python 中执行：
```python
from memory_v2 import get_memory
memory = get_memory("./data/agent_memory.db")
data = memory.export_user_data("user_id")
```

---

## 📝 更新日志

### v2.0 (当前版本)
- ✅ 添加用户认证系统
- ✅ 添加管理后台
- ✅ SQLite 持久化存储
- ✅ 严格用户隔离
- ✅ 会话历史搜索
- ✅ 用户画像自动学习

---

## 🔐 安全建议

1. **首次登录后立即修改 admin 密码**
2. **生产环境部署时**：
   - 修改 `JWT_SECRET` 环境变量
   - 启用 HTTPS
   - 定期备份数据库
   - 限制管理后台访问 IP

3. **用户密码要求**：
   - 至少 6 位字符
   - 建议包含字母+数字
   - 定期更换密码

---

## 📞 技术支持

如有问题，请查看：
- 数据库文件：`./data/agent_memory.db`
- 日志输出：控制台
- 前端代码：`./templates/`
- 后端代码：`app.py`, `auth.py`, `memory_v2.py`
