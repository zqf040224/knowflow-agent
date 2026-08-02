# NAS 存储接入测试说明

本文说明如何让 Agent 使用已挂载的 NAS 目录保存上传文件、知识库索引、表格结构库和后台备份。

## 1. 推荐目录结构

在 NAS 共享目录里准备一个 Agent 专用目录，例如挂载到应用服务器：

```text
/mnt/knowledge-agent
  uploads/
  knowledge_base/
  outputs/
  backups/
  知识库/
```

说明：

- `uploads/`：用户上传的临时文件和入库原文件。
- `knowledge_base/`：FAISS 索引、索引元数据、入库 manifest、表格结构库。
- `outputs/`：预留给后续导出文件落盘使用。
- `knowledge_base/admin_backups/`：后台删除、重建、改元数据前的索引备份。也可以用 `ADMIN_BACKUP_DIR` 指到 `/mnt/knowledge-agent/backups`。
- `知识库/`：原始知识库部门目录，用于后台上传分类和账号部门白名单。

## 2. 环境变量

本地默认不用改。测试 NAS 时在 `.env` 中配置：

```env
STORAGE_BACKEND=nas
NAS_MOUNT_PATH=/mnt/knowledge-agent
```

默认会自动使用：

```text
UPLOADS_DIR=/mnt/knowledge-agent/uploads
KNOWLEDGE_BASE_DIR=/mnt/knowledge-agent/knowledge_base
OUTPUTS_DIR=/mnt/knowledge-agent/outputs
KNOWLEDGE_SOURCE_DIR=/mnt/knowledge-agent/知识库
INGESTION_MANIFEST_DB=/mnt/knowledge-agent/knowledge_base/ingestion_manifest.sqlite
SPREADSHEET_DB_PATH=/mnt/knowledge-agent/knowledge_base/spreadsheets.sqlite
ADMIN_BACKUP_DIR=/mnt/knowledge-agent/knowledge_base/admin_backups
```

如果公司 NAS 的目录名称不同，可以单独覆盖任意路径：

```env
UPLOADS_DIR=/mnt/company-nas/knowledge-agent/uploads
KNOWLEDGE_BASE_DIR=/mnt/company-nas/knowledge-agent/vector-index
KNOWLEDGE_SOURCE_DIR=/mnt/company-nas/knowledge-agent/source-docs
ADMIN_BACKUP_DIR=/mnt/company-nas/knowledge-agent/backups
```

## 3. 启动前检查

在应用服务器上确认运行 Agent 的系统账号可以读写：

```bash
mkdir -p /mnt/knowledge-agent/uploads /mnt/knowledge-agent/knowledge_base /mnt/knowledge-agent/outputs
touch /mnt/knowledge-agent/uploads/.write_test
rm /mnt/knowledge-agent/uploads/.write_test
```

如果这里失败，Agent 也会失败。先修 NAS 挂载权限。

## 4. 健康检查

启动服务后访问：

```text
GET /api/health
```

管理员登录后也可以访问：

```text
GET /api/admin/storage-health
```

返回里的 `storage.ok` 为 `true`，表示 `uploads`、`knowledge_base`、`outputs`、`admin_backups` 均可写。

## 5. 注意事项

- NAS 权限建议只给 Agent 服务账号读写，普通用户不要直接访问受控目录。
- FAISS 文件可以放 NAS，但不要多进程同时写同一套索引。
- SQLite 放 NAS 能用于测试，但生产长期建议把用户、会话、审计等数据迁到 PostgreSQL/MySQL。
- 更新向量库时建议先在临时目录生成新索引，校验通过后再切换，避免读写冲突。
