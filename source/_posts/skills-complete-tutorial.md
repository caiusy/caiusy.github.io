---
title: Skills 完整开发教程 - 从零开始掌握 MCP Skills 开发
date: 2026-03-04 21:27:28
categories:
  - 技术教程
  - AI开发
tags:
  - MCP
  - Skills
  - Claude
  - 阿里千问
  - Python
  - 教程
cover: /images/skills-tutorial/01_mcp_architecture.png
description: 从零开始掌握 MCP Skills 开发的完整教程，以阿里千问为例，包含核心概念、代码实现、高级优化和发布实战。
---

# Skills 完整开发教程

> 从零开始掌握 MCP Skills 开发 - 以阿里千问为例

---

## 目录

- [第一部分：核心概念](#第一部分核心概念)
- [第二部分：代码实现](#第二部分代码实现)
- [第三部分：高级优化](#第三部分高级优化)
- [第四部分：发布与实战](#第四部分发布与实战)

---

# 第一部分：核心概念

## 一、什么是 Skill？用人话说

想象一下：

**没有 Skill 的 Claude：**

- 我只能聊天、读文件、写代码
- 就像一个只会基础操作的助手

**有了 Skill 的 Claude：**

- 我可以调用外部 API（比如阿里千问、百度文心）
- 我可以操作数据库
- 我可以发送邮件、发微信
- 我可以生成图片、处理视频
- 就像给我装上了各种"插件"

**类比：**

- Claude 本体 = 手机系统
- Skills = 手机 App
- MCP = App Store 的安装协议

---

## 二、MCP 协议到底是什么？

### 2.1 传统方式的问题

假设你想让 Claude 调用阿里千问 API：

```
❌ 传统方式（不可行）：
用户: "帮我调用千问 API"
Claude: "我不知道怎么调用，我没有网络访问权限"
```

### 2.2 MCP 的解决方案

```
✅ MCP 方式：
用户: "帮我调用千问 API"
Claude: "我看到有一个 qwen-chat 工具可用"
       ↓ 调用 MCP Server
MCP Server: 执行 Python 代码 → 调用千问 API → 返回结果
Claude: "千问说：[结果内容]"
```

### 2.3 MCP 的三个核心组件

![MCP 架构图](/images/skills-tutorial/01_mcp_architecture.png)

**架构说明：**

1. **Claude (AI 助手)** - 我，负责理解用户意图和调用工具
2. **MCP Server** - 本地运行的 Python/Node 程序，负责执行具体操作
3. **外部 API** - 阿里千问、数据库、文件系统等外部服务

**通信方式：**

- Claude ↔ MCP Server：通过 MCP 协议（JSON-RPC 2.0）
- MCP Server ↔ 外部 API：通过 HTTP/SQL 等标准协议

---

## 三、MCP Server 的工作原理

### 3.1 启动过程

```bash
# 1. Kiro 读取配置文件
~/.kiro/settings/mcp.json

# 2. 启动 MCP Server 进程
python /path/to/server.py

# 3. 建立通信管道（stdio）
Kiro ←→ MCP Server
     (通过标准输入输出通信)
```

### 3.2 通信协议

MCP 使用 **JSON-RPC 2.0** 协议：

```json
// Claude 发送请求
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "qwen_chat",
    "arguments": {
      "prompt": "你好"
    }
  }
}

// MCP Server 返回响应
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "你好！有什么我可以帮助你的吗？"
      }
    ]
  }
}
```

### 3.3 完整的调用流程

![通信流程图](/images/skills-tutorial/02_communication_flow.png)

**流程详解：**

1. **用户输入**："用千问帮我写首诗"
2. **Claude 识别意图**：需要调用 qwen_chat 工具
3. **Claude 构造请求**：包含工具名称和参数
4. **Kiro 转发**：通过 stdio 发送给 MCP Server
5. **MCP Server 执行**：运行 Python 代码
6. **调用千问 API**：发送 HTTP 请求
7. **返回结果**：MCP Server → Kiro → Claude
8. **Claude 整理**：格式化输出
9. **展示给用户**："千问创作了这首诗..."

---

## 四、Skill 的文件结构详解

### 4.1 最小化 Skill 结构

```
my-qwen-skill/
├── server.py          # 核心：MCP Server 代码
└── requirements.txt   # 依赖包列表
```

### 4.2 完整 Skill 结构

![文件结构图](/images/skills-tutorial/03_file_structure.png)

```
qwen-skill/
├── server.py           # MCP Server 主代码
├── qwen_api.py         # 千问 API 封装
├── config.py           # 配置管理
├── requirements.txt    # 依赖
└── tests/
    └── test_api.py     # 测试代码
```

### 4.3 各文件的作用

#### server.py - 核心代码

```python
# 这个文件做三件事：

# 1. 注册工具（告诉 Claude 有哪些功能）
@app.list_tools()
async def list_tools():
    return [Tool(name="qwen_chat", ...)]

# 2. 执行工具（实际调用 API）
@app.call_tool()
async def call_tool(name, arguments):
    if name == "qwen_chat":
        return await call_qwen_api(arguments)

# 3. 启动服务（监听 Claude 的请求）
if __name__ == "__main__":
    mcp.server.stdio.run(app)
```

---

## 五、配置文件详解

### 5.1 MCP 配置文件位置

```bash
# 用户级配置（全局）
~/.kiro/settings/mcp.json

# 工作区配置（项目级）
/path/to/project/.kiro/settings/mcp.json
```

### 5.2 配置文件结构

![配置文件结构](/images/skills-tutorial/04_config_structure.png)

```json
{
  "mcpServers": {
    "qwen-chat": {
      "command": "python",
      "args": ["/Users/caius/skills/qwen/server.py"],
      "env": {
        "DASHSCOPE_API_KEY": "sk-xxx"
      },
      "disabled": false,
      "autoApprove": ["qwen_chat", "qwen_vision"]
    }
  }
}
```

### 5.3 配置项详解


| 配置项           | 说明              | 示例                            |
| ------------- | --------------- | ----------------------------- |
| `command`     | 启动 Server 的命令   | `"python"`, `"node"`, `"uvx"` |
| `args`        | 传给命令的参数         | `["server.py"]`               |
| `env`         | 环境变量（API Key 等） | `{"API_KEY": "xxx"}`          |
| `disabled`    | 是否禁用此 Server    | `false`                       |
| `autoApprove` | 自动批准的工具列表       | `["tool1", "tool2"]`          |


---

## 六、为什么需要 MCP？

### 6.1 安全性

```python
# ❌ 不安全：直接在 Claude 中执行
# Claude 无法直接运行任意代码

# ✅ 安全：通过 MCP Server
# - Server 在本地运行，用户可控
# - API Key 存在本地，不会泄露
# - 可以审计所有调用
```

### 6.2 扩展性

```python
# 一个 MCP Server 可以提供多个工具
@app.list_tools()
async def list_tools():
    return [
        Tool(name="qwen_chat"),      # 文本对话
        Tool(name="qwen_vision"),    # 图像理解
        Tool(name="qwen_code"),      # 代码生成
        Tool(name="qwen_translate"), # 翻译
    ]
```

### 6.3 标准化

```
所有 MCP Server 使用相同的协议：
- Python 实现：FastMCP / mcp
- Node.js 实现：@modelcontextprotocol/sdk
- 其他语言：只要实现 JSON-RPC 2.0 即可
```

---

## 七、常见误区

### 误区 1：MCP Server 需要网络服务器

```
❌ 错误理解：
"我需要启动一个 HTTP 服务器，监听 8080 端口"

✅ 正确理解：
"MCP Server 通过 stdio（标准输入输出）通信，
不需要网络端口，就像命令行程序一样"
```

### 误区 2：每次调用都要重启 Server

```
❌ 错误理解：
"每次调用工具，都要重新启动 Python 进程"

✅ 正确理解：
"MCP Server 启动一次后保持运行，
可以处理多次工具调用，直到 Kiro 关闭"
```

### 误区 3：Skill 和 MCP Server 是两个东西

```
❌ 错误理解：
"Skill 是一个东西，MCP Server 是另一个东西"

✅ 正确理解：
"Skill 就是 MCP Server + 配置文件的组合，
Skill 是用户视角的名称，
MCP Server 是技术实现"
```

---

# 第二部分：代码实现

## 一、从零开始写 MCP Server

### 1.1 最简单的 MCP Server

```python
#!/usr/bin/env python3
"""
最简单的 MCP Server - Hello World
"""
from mcp.server import Server
from mcp.types import Tool, TextContent

# 创建 Server 实例
app = Server("hello-world")

# 注册工具列表
@app.list_tools()
async def list_tools() -> list[Tool]:
    """告诉 Claude 有哪些工具可用"""
    return [
        Tool(
            name="say_hello",
            description="向用户打招呼",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "用户的名字"
                    }
                },
                "required": ["name"]
            }
        )
    ]

# 处理工具调用
@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """执行工具调用"""
    if name == "say_hello":
        user_name = arguments.get("name", "朋友")
        return [TextContent(
            type="text",
            text=f"你好，{user_name}！欢迎使用 MCP！"
        )]
    
    raise ValueError(f"未知工具: {name}")

# 启动 Server
if __name__ == "__main__":
    import mcp.server.stdio
    mcp.server.stdio.run(app)
```

**这个代码做了什么？**

1. **创建 Server**：`Server("hello-world")` - 给 Server 起个名字
2. **注册工具**：`@app.list_tools()` - 告诉 Claude 有 `say_hello` 工具
3. **处理调用**：`@app.call_tool()` - 当 Claude 调用工具时执行
4. **启动服务**：`mcp.server.stdio.run(app)` - 开始监听请求

---

## 二、阿里千问 Skill 完整实现

### 2.1 项目结构

```bash
qwen-skill/
├── server.py           # MCP Server 主代码
├── qwen_api.py         # 千问 API 封装
├── config.py           # 配置管理
├── requirements.txt    # 依赖
└── tests/
    └── test_api.py     # 测试代码
```

### 2.2 依赖文件 (requirements.txt)

```txt
mcp>=0.9.0
httpx>=0.27.0
pydantic>=2.0.0
python-dotenv>=1.0.0
```

### 2.3 配置管理 (config.py)

```python
"""配置管理模块"""
import os
from typing import Optional
from pydantic import BaseModel, Field

class QwenConfig(BaseModel):
    """千问配置"""
    api_key: str = Field(..., description="DashScope API Key")
    base_url: str = Field(
        default="https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
        description="API 基础 URL"
    )
    default_model: str = Field(
        default="qwen-max",
        description="默认模型"
    )
    timeout: int = Field(
        default=30,
        description="请求超时时间（秒）"
    )

def load_config() -> QwenConfig:
    """从环境变量加载配置"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("未设置 DASHSCOPE_API_KEY 环境变量")
    
    return QwenConfig(
        api_key=api_key,
        base_url=os.getenv("QWEN_BASE_URL", QwenConfig.base_url),
        default_model=os.getenv("QWEN_DEFAULT_MODEL", "qwen-max"),
        timeout=int(os.getenv("QWEN_TIMEOUT", "30"))
    )
```

### 2.4 API 封装 (qwen_api.py)

```python
"""千问 API 封装"""
import httpx
from typing import Optional, AsyncIterator
from config import QwenConfig

class QwenAPI:
    """千问 API 客户端"""
    
    def __init__(self, config: QwenConfig):
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=config.timeout,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json"
            }
        )
    
    async def chat(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None
    ) -> str:
        """文本对话"""
        model = model or self.config.default_model
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "input": {"messages": messages},
            "parameters": {"result_format": "message"}
        }
        
        try:
            response = await self.client.post(
                self.config.base_url,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            return result["output"]["choices"][0]["message"]["content"]
            
        except httpx.HTTPStatusError as e:
            raise Exception(f"API 请求失败 ({e.response.status_code}): {e.response.text}")
        except KeyError as e:
            raise Exception(f"API 响应格式错误: {e}")
        except Exception as e:
            raise Exception(f"未知错误: {e}")
    
    async def close(self):
        """关闭客户端"""
        await self.client.aclose()
```

### 2.5 MCP Server 主代码 (server.py)

```python
#!/usr/bin/env python3
"""阿里千问 MCP Server"""
import asyncio
from typing import Any
from mcp.server import Server
from mcp.types import Tool, TextContent, Resource
from config import load_config
from qwen_api import QwenAPI

# 初始化
app = Server("qwen-chat")
config = load_config()
qwen = QwenAPI(config)

@app.list_tools()
async def list_tools() -> list[Tool]:
    """注册工具"""
    return [
        Tool(
            name="qwen_chat",
            description="使用阿里千问进行文本对话",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "对话内容"},
                    "model": {
                        "type": "string",
                        "enum": ["qwen-max", "qwen-plus", "qwen-turbo"],
                        "default": "qwen-max"
                    }
                },
                "required": ["prompt"]
            }
        ),
        Tool(
            name="qwen_vision",
            description="使用千问理解图像内容",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_url": {"type": "string"},
                    "question": {"type": "string"}
                },
                "required": ["image_url", "question"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """处理工具调用"""
    try:
        if name == "qwen_chat":
            result = await qwen.chat(
                prompt=arguments["prompt"],
                model=arguments.get("model")
            )
            return [TextContent(type="text", text=f"千问回复：\n\n{result}")]
        
        elif name == "qwen_vision":
            result = await qwen.vision(
                image_url=arguments["image_url"],
                question=arguments["question"]
            )
            return [TextContent(type="text", text=f"千问视觉分析：\n\n{result}")]
    
    except Exception as e:
        return [TextContent(type="text", text=f"❌ 调用失败: {str(e)}")]

if __name__ == "__main__":
    import mcp.server.stdio
    mcp.server.stdio.run(app)
```

---

## 三、安装和配置

### 3.1 创建项目

```bash
# 1. 创建目录
mkdir -p ~/skills/qwen-skill
cd ~/skills/qwen-skill

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # macOS/Linux

# 3. 安装依赖
pip install mcp httpx pydantic python-dotenv
```

### 3.2 配置 API Key

```bash
# 方式 1：环境变量（推荐）
export DASHSCOPE_API_KEY="sk-your-api-key-here"

# 方式 2：.env 文件
echo "DASHSCOPE_API_KEY=sk-your-api-key-here" > .env
```

### 3.3 配置 MCP

编辑 `~/.kiro/settings/mcp.json`：

```json
{
  "mcpServers": {
    "qwen-chat": {
      "command": "/Users/caius/skills/qwen-skill/venv/bin/python",
      "args": ["/Users/caius/skills/qwen-skill/server.py"],
      "env": {
        "DASHSCOPE_API_KEY": "sk-your-api-key-here"
      },
      "disabled": false,
      "autoApprove": ["qwen_chat", "qwen_vision"]
    }
  }
}
```

**注意事项：**

- `command` 使用虚拟环境的 Python 路径
- `args` 使用绝对路径
- `env` 中配置 API Key

### 3.4 重启 MCP Server

在 Kiro 中：

1. 按 `Cmd+Shift+P`（macOS）或 `Ctrl+Shift+P`（Windows/Linux）
2. 输入 "MCP"
3. 选择 "Reconnect MCP Servers"

---

## 四、测试和调试

### 4.1 手动测试

```python
# test_manual.py
import asyncio
from config import load_config
from qwen_api import QwenAPI

async def main():
    config = load_config()
    qwen = QwenAPI(config)
    
    # 测试对话
    result = await qwen.chat("写一首关于春天的诗")
    print("对话结果：", result)
    
    await qwen.close()

if __name__ == "__main__":
    asyncio.run(main())
```

运行：

```bash
python test_manual.py
```

### 4.2 MCP Inspector 调试

```bash
# 安装 MCP Inspector
npm install -g @modelcontextprotocol/inspector

# 启动调试
npx @modelcontextprotocol/inspector python /path/to/server.py
```

这会打开一个 Web 界面，可以：

- 查看注册的工具
- 手动调用工具
- 查看请求/响应
- 调试错误

---

## 五、常见问题和解决方案

### 问题 1：Server 无法启动

```bash
# 症状
[ERROR] Failed to start MCP server: qwen-chat

# 排查步骤
# 1. 检查 Python 路径
which python
/Users/caius/skills/qwen-skill/venv/bin/python

# 2. 检查脚本路径
ls -la /Users/caius/skills/qwen-skill/server.py

# 3. 手动运行测试
python /Users/caius/skills/qwen-skill/server.py
# 应该看到 Server 启动，等待输入

# 4. 检查依赖
pip list | grep mcp
```

### 问题 2：API Key 未生效

```bash
# 症状
❌ 调用失败: API 请求失败 (401): Unauthorized

# 解决方案
# 1. 检查环境变量
echo $DASHSCOPE_API_KEY

# 2. 在 mcp.json 中明确设置
{
  "env": {
    "DASHSCOPE_API_KEY": "sk-实际的key"
  }
}
```

### 问题 3：工具调用超时

```python
# 症状
❌ 调用失败: Request timeout

# 解决方案：增加超时时间
config = QwenConfig(
    api_key=api_key,
    timeout=60  # 增加到 60 秒
)
```

---

# 第三部分：高级优化

## 一、缓存机制

### 1.1 为什么需要缓存？

```python
# 问题场景
用户: "用千问翻译 hello"
→ 调用 API，耗时 2 秒，花费 0.01 元

用户: "再翻译一次 hello"  # 相同请求
→ 又调用 API，耗时 2 秒，又花费 0.01 元

# 使用缓存后
用户: "用千问翻译 hello"
→ 调用 API，耗时 2 秒，花费 0.01 元，存入缓存

用户: "再翻译一次 hello"
→ 从缓存读取，耗时 0.01 秒，花费 0 元 ✓
```

### 1.2 简单内存缓存

```python
"""简单缓存实现"""
import hashlib
import json

class CachedQwenAPI(QwenAPI):
    """带缓存的千问 API"""
    
    def __init__(self, config: QwenConfig):
        super().__init__(config)
        self._cache = {}
    
    def _make_cache_key(self, prompt: str, model: str, system: str = None) -> str:
        """生成缓存键"""
        data = {"prompt": prompt, "model": model, "system": system}
        content = json.dumps(data, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    async def chat(self, prompt: str, model: str = None, system: str = None) -> str:
        """带缓存的对话"""
        model = model or self.config.default_model
        cache_key = self._make_cache_key(prompt, model, system)
        
        # 检查缓存
        if cache_key in self._cache:
            print(f"[缓存命中] {cache_key[:8]}...")
            return self._cache[cache_key]
        
        # 调用 API
        result = await super().chat(prompt, model, system)
        
        # 存入缓存
        self._cache[cache_key] = result
        return result
```

### 1.3 持久化缓存（使用 SQLite）

```python
"""持久化缓存"""
import sqlite3
import time
from pathlib import Path

class PersistentCache:
    """SQLite 持久化缓存"""
    
    def __init__(self, db_path: str = "~/.kiro/cache/qwen.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                expires_at INTEGER
            )
        """)
        conn.commit()
        conn.close()
    
    def get(self, key: str) -> Optional[str]:
        """获取缓存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT value, expires_at FROM cache WHERE key = ?",
            (key,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        value, expires_at = row
        
        # 检查是否过期
        if expires_at and time.time() > expires_at:
            self.delete(key)
            return None
        
        return value
    
    def set(self, key: str, value: str, ttl: int = 3600):
        """设置缓存（默认1小时过期）"""
        conn = sqlite3.connect(self.db_path)
        expires_at = int(time.time() + ttl) if ttl else None
        
        conn.execute(
            """
            INSERT OR REPLACE INTO cache (key, value, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (key, value, int(time.time()), expires_at)
        )
        conn.commit()
        conn.close()
```

---

## 二、错误处理和重试

### 2.1 重试装饰器

```python
"""重试机制"""
import asyncio
from functools import wraps

def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """重试装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    
                    print(f"[重试] 第 {attempt + 1}/{max_attempts} 次失败: {e}")
                    print(f"[重试] 等待 {current_delay:.1f} 秒后重试...")
                    
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
        
        return wrapper
    return decorator

# 使用重试
class QwenAPI:
    @retry(max_attempts=3, delay=1.0, backoff=2.0)
    async def chat(self, prompt: str, model: str = None) -> str:
        """带重试的对话"""
        # ... API 调用代码
```

### 2.2 详细的错误处理

```python
"""完善的错误处理"""

class QwenError(Exception):
    """千问 API 错误基类"""
    pass

class QwenAPIError(QwenError):
    """API 调用错误"""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"API Error ({status_code}): {message}")

class QwenRateLimitError(QwenError):
    """速率限制错误"""
    pass

class QwenTimeoutError(QwenError):
    """超时错误"""
    pass

# 在 MCP Server 中处理错误
@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    try:
        if name == "qwen_chat":
            result = await qwen.chat(...)
            return [TextContent(type="text", text=result)]
    
    except QwenRateLimitError as e:
        return [TextContent(
            type="text",
            text=f"⚠️ {str(e)}\n\n建议：等待 1 分钟后重试"
        )]
    
    except QwenAPIError as e:
        return [TextContent(
            type="text",
            text=f"❌ API 错误: {str(e)}\n\n请检查配置和网络连接"
        )]
    
    except QwenTimeoutError as e:
        return [TextContent(
            type="text",
            text=f"⏱️ {str(e)}\n\n建议：使用更快的模型或增加超时时间"
        )]
```

---

## 三、日志记录

### 3.1 配置日志

```python
"""日志配置"""
import logging
from pathlib import Path

def setup_logging(log_dir: str = "~/.kiro/logs"):
    """配置日志"""
    log_dir = Path(log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建 logger
    logger = logging.getLogger("qwen-skill")
    logger.setLevel(logging.DEBUG)
    
    # 文件处理器（详细日志）
    file_handler = logging.FileHandler(
        log_dir / "qwen-skill.log",
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    
    # 控制台处理器（简要日志）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 使用日志
logger = setup_logging()
```

### 3.2 在代码中使用日志

```python
class QwenAPI:
    def __init__(self, config: QwenConfig):
        self.config = config
        self.logger = logging.getLogger("qwen-skill.api")
    
    async def chat(self, prompt: str, model: str = None) -> str:
        model = model or self.config.default_model
        
        self.logger.info(f"调用千问 API: model={model}, prompt_len={len(prompt)}")
        self.logger.debug(f"完整请求: prompt={prompt[:100]}...")
        
        try:
            response = await self.client.post(...)
            result = response.json()
            
            self.logger.info(f"API 调用成功: response_len={len(result)}")
            return result
        
        except Exception as e:
            self.logger.error(f"API 调用失败: {e}", exc_info=True)
            raise
```

---

## 四、性能优化

### 4.1 连接池

```python
"""使用连接池优化性能"""
import httpx

class QwenAPI:
    def __init__(self, config: QwenConfig):
        self.config = config
        
        # 配置连接池
        limits = httpx.Limits(
            max_keepalive_connections=10,  # 最大保持连接数
            max_connections=20,            # 最大连接数
            keepalive_expiry=30.0          # 连接保持时间
        )
        
        self.client = httpx.AsyncClient(
            timeout=config.timeout,
            limits=limits,
            headers={...}
        )
```

### 4.2 并发控制

```python
"""控制并发请求数"""
import asyncio

class QwenAPI:
    def __init__(self, config: QwenConfig):
        # ...
        self.semaphore = asyncio.Semaphore(5)  # 最多 5 个并发请求
    
    async def chat(self, prompt: str, model: str = None) -> str:
        async with self.semaphore:
            # 限制并发数
            return await self._do_chat(prompt, model)
```

### 4.3 批量处理

```python
"""批量处理多个请求"""
async def batch_chat(qwen: QwenAPI, prompts: list[str]) -> list[str]:
    """批量对话"""
    tasks = [qwen.chat(prompt) for prompt in prompts]
    
    # 并发执行所有请求
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 处理结果
    outputs = []
    for result in results:
        if isinstance(result, Exception):
            outputs.append(f"错误: {result}")
        else:
            outputs.append(result)
    
    return outputs
```

---

# 第四部分：发布与实战

![开发工作流](/images/skills-tutorial/05_workflow.png)

## 一、完整的项目结构

### 1.1 生产级目录结构

```
qwen-skill/
├── src/
│   ├── __init__.py
│   ├── server.py          # MCP Server 入口
│   ├── api.py             # API 封装
│   ├── config.py          # 配置管理
│   ├── cache.py           # 缓存实现
│   └── errors.py          # 错误定义
├── tests/
│   ├── test_api.py
│   └── test_server.py
├── docs/
│   └── README.md
├── requirements.txt       # 生产依赖
├── setup.py              # 安装脚本
└── README.md
```

### 1.2 setup.py 安装脚本

```python
from setuptools import setup, find_packages

setup(
    name="qwen-mcp-skill",
    version="1.0.0",
    author="Your Name",
    description="阿里千问 MCP Skill",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "mcp>=0.9.0",
        "httpx>=0.27.0",
        "pydantic>=2.0.0",
    ],
)
```

---

## 二、发布到 GitHub

### 2.1 初始化 Git 仓库

```bash
cd qwen-skill
git init
git add .
git commit -m "Initial commit: Qwen MCP Skill v1.0.0"
```

### 2.2 推送到 GitHub

```bash
git remote add origin https://github.com/yourusername/qwen-mcp-skill.git
git branch -M main
git push -u origin main
```

---

## 三、发布到 PyPI

```bash
# 安装发布工具
pip install build twine

# 构建包
python -m build

# 上传到 PyPI
twine upload dist/*
```

---

## 四、实战案例

### 案例 1：多模型对比工具

对比不同模型的回答质量，帮助用户选择最合适的模型。

### 案例 2：智能摘要工具

自动抓取网页内容并生成摘要，支持短、中、长三种长度。

### 案例 3：代码审查助手

自动审查代码质量，提供改进建议。

---

## 五、最佳实践

### 5.1 安全性

- 使用环境变量存储 API Key
- 验证用户输入
- 限制请求速率

### 5.2 可维护性

- 使用类型注解
- 编写文档字符串
- 添加单元测试

### 5.3 用户体验

- 提供清晰的错误信息
- 显示进度提示
- 提供使用示例

---

## 六、总结

恭喜！你现在已经完全掌握了 Skills 开发：

1. ✅ 理解 MCP 协议原理
2. ✅ 编写完整的 MCP Server
3. ✅ 实现缓存、重试、日志等功能
4. ✅ 发布到 GitHub 和 PyPI
5. ✅ 掌握最佳实践

**下一步建议：**

- 尝试创建自己的 Skill
- 集成其他 AI 服务（百度文心、讯飞星火等）
- 分享你的 Skill 给社区

---

**作者**: Caius