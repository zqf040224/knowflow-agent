# NAS 对接技术方案（存档）

**文档版本**：v1.0
**创建日期**：2025-04-11
**适用项目**：智能知识库平台专属桌面 Agent
**技术栈**：Python + smbprotocol + PyQt6

---

## 一、方案概述

### 1.1 方案定位

本方案采用**混合架构**：
- **核心知识库**：离线构建 FAISS 索引，打包进 Agent（公文模板、制度文件、常用范文等 500-1000 份）
- **NAS 文件浏览**：仅读取元数据（文件列表），不下载文件内容
- **按需处理**：用户拖拽文件时临时下载，用完即删

### 1.2 核心优势

| 优势 | 说明 |
|------|------|
| 存储占用小 | 仅核心知识库（50-100MB），NAS 文件不落地 |
| 检索速度快 | 核心文档本地检索 < 0.1 秒 |
| 实时更新 | NAS 新文件立即可查（浏览 + 拖拽） |
| 运维简单 | 管理员每月更新 1 次核心知识库 |
| 技术难度低 | NAS 仅读取元数据，无需同步文件 |

---

## 二、技术架构

### 2.1 整体架构

```
【Agent 客户端】
┌─────────────────────────────────────────────┐
│  主界面                                      │
├──────────────┬──────────────────────────────┤
│  左侧边栏    │  主内容区                      │
│              │  ┌──────────────────────┐    │
│  🔍 智能检索  │  │  【模式 1: 智能检索】   │    │
│  (本地 FAISS) │  │  用户输入关键词        │    │
│              │  │  → 检索本地知识库     │    │
│  📂 NAS 浏览  │  └──────────────────────┘    │
│  (SMB 协议)   │                              │
│              │  ┌──────────────────────┐    │
│  📁 公共资料  │  │  【模式 2: NAS 浏览】   │    │
│  📁 行政部    │  │  显示 NAS 文件列表      │    │
│  📁 照片库    │  │  → 双击预览           │    │
│              │  │  → 拖拽到聊天框处理   │    │
│              │  └──────────────────────┘    │
└──────────────┴──────────────────────────────┘
```

### 2.2 模块划分

```
【NAS 对接模块】
├─ login.py              # NAS 登录 + 会话管理
├─ permission.py         # 权限检测（登录时一次性）
├─ browser.py            # NAS 文件浏览（仅元数据）
├─ preview.py            # 文件预览（临时下载）
├─ drag_drop.py          # 拖拽功能（临时下载解析）
└─ cache.py              # 临时缓存管理（自动清理）

【核心知识库模块】
├─ builder.py            # 知识库构建工具（管理员用）
├─ faiss_index.py        # FAISS 索引管理
└─ retrieval.py          # 本地检索
```

---

## 三、NAS 对接实现

### 3.1 环境准备

#### 3.1.1 依赖安装

```bash
# 核心依赖
pip install smbprotocol
pip install PyQt6
pip install FAISS-cpu
pip install langchain
pip install dashscope

# 文档解析
pip install PyPDF2
pip install python-docx
pip install chardet
```

#### 3.1.2 NAS 配置要求

- **NAS 品牌**：群晖（Synology）
- **协议**：SMB 2.x / SMB 3.x
- **端口**：445（默认）
- **权限**：
  - 公共资料文件夹：所有人可读
  - 部门文件夹：仅本部门可读
  - 照片库：所有人可读

---

### 3.2 NAS 登录模块

#### 3.2.1 登录弹窗（PyQt6）

```python
# login.py
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLabel,
                             QLineEdit, QPushButton, QMessageBox)
from PyQt6.QtCore import Qt
import smbclient

class NASLoginDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.session = None

    def init_ui(self):
        """初始化登录界面"""
        self.setWindowTitle("NAS 账号登录")
        self.setModal(True)
        self.setFixedSize(400, 250)

        layout = QVBoxLayout()

        # NAS 地址
        layout.addWidget(QLabel("NAS 地址:"))
        self.nas_address = QLineEdit()
        self.nas_address.setPlaceholderText("例如：192.0.2.10")
        self.nas_address.setText("192.0.2.10")  # 文档保留示例地址
        layout.addWidget(self.nas_address)

        # 用户名
        layout.addWidget(QLabel("NAS 用户名:"))
        self.username = QLineEdit()
        self.username.setPlaceholderText("例如：demo-user@示例部门甲")
        layout.addWidget(self.username)

        # 密码
        layout.addWidget(QLabel("密码:"))
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.password)

        # 按钮
        btn_layout = QVBoxLayout()

        self.login_btn = QPushButton("登录")
        self.login_btn.clicked.connect(self.do_login)
        btn_layout.addWidget(self.login_btn)

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def do_login(self):
        """执行登录校验"""
        nas_ip = self.nas_address.text()
        username = self.username.text()
        password = self.password.text()

        if not username or not password:
            QMessageBox.warning(self, "提示", "请输入用户名和密码")
            return

        try:
            # 建立 SMB 会话
            self.session = smbclient.register_session(
                server=nas_ip,
                username=username,
                password=password
            )

            # 测试连接
            smbclient.listdir(rf"\\{nas_ip}\公共资料")

            # 登录成功
            self.accept()

        except smbprotocol.exceptions.AccessDenied:
            QMessageBox.critical(self, "错误", "NAS 账号或密码错误")
        except smbprotocol.exceptions.ConnectionError as e:
            QMessageBox.critical(self, "错误", f"无法连接 NAS：{str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"登录失败：{str(e)}")

    def get_session(self):
        """返回 SMB 会话"""
        return self.session

    def get_user_info(self):
        """返回用户信息"""
        return {
            "username": self.username.text(),
            "nas_ip": self.nas_address.text()
        }
```

#### 3.2.2 会话管理

```python
# session_manager.py
import smbclient
from typing import Optional, Dict

class SessionManager:
    """SMB 会话管理器"""

    _instance = None
    _sessions: Dict[str, smbclient.SMBSession] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def create_session(self, server: str, username: str, password: str) -> smbclient.SMBSession:
        """创建新的 SMB 会话"""
        session = smbclient.register_session(
            server=server,
            username=username,
            password=password
        )
        self._sessions[f"{server}_{username}"] = session
        return session

    def get_session(self, server: str, username: str) -> Optional[smbclient.SMBSession]:
        """获取已存在的会话"""
        return self._sessions.get(f"{server}_{username}")

    def remove_session(self, server: str, username: str):
        """移除会话"""
        key = f"{server}_{username}"
        if key in self._sessions:
            del self._sessions[key]

    def clear_all(self):
        """清空所有会话"""
        self._sessions.clear()
```

---

### 3.3 权限检测模块

#### 3.3.1 权限判断逻辑

```python
# permission.py
import smbclient
from typing import List, Dict

class PermissionChecker:
    """NAS 权限检测器"""

    def __init__(self, session: smbclient.SMBSession, nas_ip: str):
        self.session = session
        self.nas_ip = nas_ip

    def check_access(self, folder_name: str) -> bool:
        """
        检查用户是否有权限访问指定文件夹
        返回：True=有权限，False=无权限
        """
        try:
            folder_path = rf"\\{self.nas_ip}\{folder_name}"
            smbclient.listdir(folder_path)
            return True
        except smbprotocol.exceptions.AccessDenied:
            return False
        except Exception:
            return False

    def get_accessible_folders(self) -> List[str]:
        """
        获取用户有权限访问的所有文件夹
        返回：文件夹名称列表
        """
        # 预定义文件夹列表
        predefined_folders = [
            "公共资料",
            "行政部",
            "财务部",
            "示例部门甲",
            "照片库"
        ]

        # 检测每个文件夹的访问权限
        accessible = []
        for folder in predefined_folders:
            if self.check_access(folder):
                accessible.append(folder)

        return accessible

    def get_user_department(self, username: str) -> str:
        """
        根据用户名判断所属部门
        例如：demo-user@示例部门甲 → 示例部门甲
        """
        if "@admin" in username:
            return "行政部"
        elif "@finance" in username:
            return "财务部"
        elif "@research" in username:
            return "示例部门甲"
        else:
            return "公共"
```

---

### 3.4 NAS 文件浏览模块

#### 3.4.1 文件列表读取（仅元数据）

```python
# browser.py
import smbclient
from datetime import datetime
from typing import List, Dict

class NASBrowser:
    """NAS 文件浏览器"""

    def __init__(self, session: smbclient.SMBSession, nas_ip: str):
        self.session = session
        self.nas_ip = nas_ip

    def list_folders(self) -> List[Dict]:
        """
        列出根目录下的文件夹（仅显示有权限的）
        """
        folders = []

        # 预定义文件夹
        predefined = ["公共资料", "示例部门甲", "示例部门乙", "照片库"]

        for folder_name in predefined:
            try:
                folder_path = rf"\\{self.nas_ip}\{folder_name}"
                # 检测是否有权限
                smbclient.listdir(folder_path)

                folders.append({
                    "name": folder_name,
                    "type": "folder",
                    "path": folder_path
                })
            except Exception:
                # 无权限则不显示
                pass

        return folders

    def list_files(self, folder_path: str) -> List[Dict]:
        """
        列出指定文件夹下的文件列表（仅元数据，不下载内容）
        """
        files = []

        try:
            for file_info in smbclient.listdir(folder_path):
                files.append({
                    "name": file_info.filename,
                    "type": "folder" if file_info.is_dir else "file",
                    "size": file_info.file_size if not file_info.is_dir else 0,
                    "modified": file_info.last_write_time,
                    "path": rf"{folder_path}\{file_info.filename}"
                })
        except Exception as e:
            print(f"读取文件列表失败：{e}")

        return files

    def get_file_metadata(self, file_path: str) -> Dict:
        """
        获取单个文件的元数据
        """
        try:
            stat = smbclient.stat(file_path)
            return {
                "name": file_path.split("\\")[-1],
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime),
                "created": datetime.fromtimestamp(stat.st_ctime)
            }
        except Exception as e:
            print(f"获取文件元数据失败：{e}")
            return {}
```

---

### 3.5 文件预览模块

#### 3.5.1 临时下载 + 预览

```python
# preview.py
import smbclient
import tempfile
import os
from pathlib import Path
from typing import Optional

class FilePreview:
    """文件预览器"""

    def __init__(self):
        self.temp_files = []  # 记录临时文件，用于清理

    def download_to_temp(self, nas_file_path: str) -> Optional[str]:
        """
        下载 NAS 文件到临时目录
        返回：临时文件路径
        """
        try:
            # 创建临时文件
            file_name = os.path.basename(nas_file_path)
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"agent_temp_{file_name}")

            # 从 NAS 下载
            with smbclient.open_file(nas_file_path, mode='rb') as nas_file:
                with open(temp_path, 'wb') as local_file:
                    local_file.write(nas_file.read())

            # 记录临时文件
            self.temp_files.append(temp_path)

            return temp_path

        except Exception as e:
            print(f"下载文件失败：{e}")
            return None

    def preview_pdf(self, file_path: str):
        """预览 PDF 文件"""
        # 使用 PyQt6 的 QPdfViewer
        from PyQt6.QtPdfWidgets import QPdfView
        from PyQt6.QtPdf import QPdfDocument

        pdf_view = QPdfView()
        document = QPdfDocument()
        document.load(file_path)
        pdf_view.setDocument(document)
        pdf_view.show()

    def preview_docx(self, file_path: str) -> str:
        """预览 Word 文档（提取文本）"""
        from docx import Document

        doc = Document(file_path)
        content = []
        for para in doc.paragraphs:
            content.append(para.text)

        return "\n".join(content)

    def preview_image(self, file_path: str):
        """预览图片"""
        from PyQt6.QtWidgets import QLabel
        from PyQt6.QtGui import QPixmap

        label = QLabel()
        pixmap = QPixmap(file_path)
        label.setPixmap(pixmap)
        label.show()

    def cleanup(self):
        """清理所有临时文件"""
        for temp_path in self.temp_files:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                print(f"清理临时文件失败：{e}")

        self.temp_files.clear()
```

---

### 3.6 拖拽功能模块

#### 3.6.1 PyQt6 拖拽实现

```python
# drag_drop.py
from PyQt6.QtWidgets import QTextEdit, QMessageBox
from PyQt6.QtCore import QMimeData
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
import os

class ChatWidget(QTextEdit):
    """支持拖拽的聊天框"""

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)  # 启用拖拽
        self.setPlaceholderText("输入需求，或拖拽文件到此处...")

    def dragEnterEvent(self, event: QDragEnterEvent):
        """拖拽进入事件"""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        """文件放下事件"""
        files = []
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if os.path.exists(file_path):
                files.append(file_path)

        if files:
            self.handle_dropped_files(files)

    def handle_dropped_files(self, file_paths: list):
        """处理拖拽的文件"""
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            self.append(f"📎 已接收文件：{file_name}")

            # 调用解析函数
            self.process_file(file_path)

    def process_file(self, file_path: str):
        """解析文件并调用大模型"""
        # 1. 根据文件类型解析内容
        if file_path.endswith(".pdf"):
            content = self.parse_pdf(file_path)
        elif file_path.endswith(".docx"):
            content = self.parse_docx(file_path)
        elif file_path.endswith(".txt"):
            content = self.parse_txt(file_path)
        else:
            self.append("⚠️ 不支持的文件格式")
            return

        # 2. 调用大模型 API
        # （此处调用 Qwen API，详见公文生成模块）
        response = self.call_llm_api(content)

        # 3. 显示结果
        self.append(f"🤖 AI 响应：{response}")

    def parse_pdf(self, file_path: str) -> str:
        """解析 PDF 文件"""
        from PyPDF2 import PdfReader

        reader = PdfReader(file_path)
        content = []
        for page in reader.pages:
            content.append(page.extract_text())

        return "\n".join(content)

    def parse_docx(self, file_path: str) -> str:
        """解析 Word 文件"""
        from docx import Document

        doc = Document(file_path)
        content = [para.text for para in doc.paragraphs]
        return "\n".join(content)

    def parse_txt(self, file_path: str) -> str:
        """解析 TXT 文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def call_llm_api(self, content: str) -> str:
        """调用大模型 API（示例）"""
        # 实际实现参考公文生成模块
        return "文件已解析，请提出您的问题..."
```

---

### 3.7 缓存管理模块

#### 3.7.1 临时缓存自动清理

```python
# cache.py
import os
import tempfile
import atexit
from typing import List

class CacheManager:
    """临时缓存管理器"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.temp_files = []
            cls._instance.register_cleanup()
        return cls._instance

    def add_temp_file(self, file_path: str):
        """添加临时文件记录"""
        self.temp_files.append(file_path)

    def remove_temp_file(self, file_path: str):
        """移除临时文件"""
        if file_path in self.temp_files:
            self.temp_files.remove(file_path)
            if os.path.exists(file_path):
                os.remove(file_path)

    def get_all_temp_files(self) -> List[str]:
        """获取所有临时文件路径"""
        return self.temp_files.copy()

    def cleanup(self):
        """清理所有临时文件"""
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"清理临时文件失败：{e}")

        self.temp_files.clear()

    def register_cleanup(self):
        """注册程序退出时的清理回调"""
        atexit.register(self.cleanup)
```

---

## 四、NAS 文件浏览器 UI 实现

### 4.1 主界面布局

```python
# main_window.py
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTreeWidget, QTreeWidgetItem,
                             QTableWidget, QTableWidgetItem, QLabel)
from PyQt6.QtCore import Qt

class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self, session, user_info):
        super().__init__()
        self.session = session
        self.user_info = user_info
        self.init_ui()

    def init_ui(self):
        """初始化主界面"""
        self.setWindowTitle("智能知识库平台 · 专属 Agent")
        self.setGeometry(100, 100, 1400, 900)

        # 中央组件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()

        # 左侧边栏：NAS 文件树
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, stretch=1)

        # 右侧：主内容区
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, stretch=3)

        central_widget.setLayout(main_layout)

    def create_left_panel(self) -> QWidget:
        """创建左侧边栏"""
        widget = QWidget()
        layout = QVBoxLayout()

        # 标题
        layout.addWidget(QLabel("📂 NAS 文件"))

        # 文件树
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabels(["名称"])

        # 加载文件夹（仅显示有权限的）
        self.load_folders()

        layout.addWidget(self.file_tree)

        widget.setLayout(layout)
        return widget

    def create_right_panel(self) -> QWidget:
        """创建右侧主内容区"""
        widget = QWidget()
        layout = QVBoxLayout()

        # 文件列表表格
        self.file_table = QTableWidget()
        self.file_table.setColumnCount(3)
        self.file_table.setHorizontalHeaderLabels(["名称", "大小", "修改时间"])

        layout.addWidget(self.file_table)

        # 聊天交互区
        self.chat_widget = ChatWidget()  # 使用前面定义的 ChatWidget
        layout.addWidget(self.chat_widget, stretch=1)

        widget.setLayout(layout)
        return widget

    def load_folders(self):
        """加载 NAS 文件夹到文件树"""
        from browser import NASBrowser

        browser = NASBrowser(self.session, self.user_info["nas_ip"])
        folders = browser.list_folders()

        for folder in folders:
            item = QTreeWidgetItem(self.file_tree)
            item.setText(0, f"📁 {folder['name']}")
            item.setData(0, Qt.ItemDataRole.UserRole, folder['path'])
```

---

## 五、开发注意事项

### 5.1 SMB 协议兼容性

```python
# 群晖 NAS 默认支持 SMB 2.x / 3.x
# 如果遇到连接问题，可尝试指定 SMB 版本：

smbclient.register_session(
    server="192.0.2.10",
    username="demo-user",
    password="******",
    require_signing=False  # 某些 NAS 需要关闭签名
)
```

### 5.2 路径处理

```python
# Windows 路径格式：\\192.0.2.10\公共资料
# Mac 路径格式：smb://192.0.2.10/公共资料

# smbprotocol 自动处理跨平台路径
# 使用 rf"\\{nas_ip}\{folder_name}" 格式即可
```

### 5.3 权限错误处理

```python
try:
    smbclient.listdir(folder_path)
except smbprotocol.exceptions.AccessDenied:
    # 无权限，不显示该文件夹
    pass
except smbprotocol.exceptions.ConnectionError:
    # 网络连接问题
    QMessageBox.critical(self, "错误", "NAS 连接中断")
```

### 5.4 临时文件清理

```python
# 方案 1：程序退出时清理
import atexit
atexit.register(cleanup_temp_files)

# 方案 2：每次关闭对话时清理
def close_event(self, event):
    cache_manager.cleanup()
    event.accept()
```

---

## 六、测试用例

### 6.1 登录测试

```python
def test_login():
    """测试 NAS 登录"""
    dialog = NASLoginDialog()
    if dialog.exec() == QDialog.DialogCode.Accepted:
        print("登录成功")
        print(f"用户信息：{dialog.get_user_info()}")
    else:
        print("登录失败或取消")
```

### 6.2 权限测试

```python
def test_permissions():
    """测试权限检测"""
    checker = PermissionChecker(session, "192.0.2.10")
    folders = checker.get_accessible_folders()
    print(f"有权限的文件夹：{folders}")
```

### 6.3 文件浏览测试

```python
def test_browse():
    """测试文件浏览"""
    browser = NASBrowser(session, "192.0.2.10")

    # 列出文件夹
    folders = browser.list_folders()
    print(f"文件夹：{folders}")

    # 列出文件
    files = browser.list_files(r"\\192.0.2.10\公共资料")
    print(f"文件：{files}")
```

---

## 七、性能优化建议

### 7.1 文件列表缓存

```python
# 缓存文件列表 5 分钟，避免频繁请求 NAS
from functools import lru_cache
import time

@lru_cache(maxsize=100)
def cached_list_files(folder_path: str, cache_time: int):
    """缓存文件列表"""
    return browser.list_files(folder_path)

# 使用：每 5 分钟刷新一次
files = cached_list_files(folder_path, int(time.time() / 300))
```

### 7.2 大文件处理

```python
# 限制预览文件大小（>50MB 的文件不预览）
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

def preview_file(file_path: str):
    metadata = browser.get_file_metadata(file_path)
    if metadata.get("size", 0) > MAX_FILE_SIZE:
        QMessageBox.warning(self, "提示", "文件过大，无法预览")
        return

    # 正常预览逻辑
```

---

## 八、常见问题

### Q1: 连接 NAS 失败

**原因**：
- NAS 地址错误
- 局域网连接问题
- SMB 协议版本不兼容

**解决**：
```python
# 1. 检查 NAS 地址是否正确
# 2. 测试局域网连通性：ping 192.0.2.10
# 3. 尝试关闭 SMB 签名
smbclient.register_session(server, username, password, require_signing=False)
```

### Q2: 权限检测不准确

**原因**：
- NAS 权限配置复杂（共享权限 + NTFS 权限）

**解决**：
```python
# 直接尝试访问文件夹，捕获 AccessDenied 异常
try:
    smbclient.listdir(folder_path)
    return True  # 有权限
except smbprotocol.exceptions.AccessDenied:
    return False  # 无权限
```

### Q3: 临时文件未清理

**原因**：
- 程序异常退出

**解决**：
```python
# 注册退出清理回调
import atexit
atexit.register(cache_manager.cleanup)

# 启动时清理旧临时文件
def cleanup_old_temp_files():
    temp_dir = tempfile.gettempdir()
    for file in os.listdir(temp_dir):
        if file.startswith("agent_temp_"):
            os.remove(os.path.join(temp_dir, file))
```

---

## 九、后续优化方向

### 9.1 文件搜索功能

```python
def search_nas_files(keyword: str) -> List[Dict]:
    """
    在 NAS 文件列表中搜索（仅搜索文件名）
    """
    results = []
    for folder in accessible_folders:
        files = browser.list_files(folder)
        for file in files:
            if keyword.lower() in file["name"].lower():
                results.append(file)
    return results
```

### 9.2 最近访问记录

```python
# 记录用户最近访问的 10 个文件
recent_files = []

def add_recent_file(file_path: str):
    recent_files.insert(0, file_path)
    if len(recent_files) > 10:
        recent_files.pop()
```

### 9.3 收藏夹功能

```python
# 用户可收藏常用文件夹
favorites = load_from_config()

def add_to_favorite(folder_path: str):
    favorites.append(folder_path)
    save_to_config(favorites)
```

---

## 十、总结

### 10.1 核心技术点

1. **SMB 协议对接**：使用 `smbprotocol` 库实现 NAS 连接
2. **权限检测**：登录时一次性检测，无权限文件夹不显示
3. **文件浏览**：仅读取元数据，不下载文件内容
4. **临时下载**：双击预览/拖拽处理时临时下载，用完即删
5. **缓存管理**：自动清理临时文件，不占用存储空间

### 10.2 开发优先级

- **Phase 1**（Week 3-4）：实现 NAS 登录、权限检测、文件浏览
- **Phase 2**（Week 5-6）：实现文件预览、拖拽功能
- **Phase 3**（后续优化）：文件搜索、最近访问、收藏夹

### 10.3 风险提示

- **网络依赖**：需确保局域网环境稳定
- **NAS 兼容性**：不同品牌 NAS 可能有差异（本方案针对群晖优化）
- **大文件处理**：>50MB 文件建议限制预览

---

**文档结束**
