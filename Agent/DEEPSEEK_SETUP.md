# DeepSeek API 配置说明

## 更新内容

本项目已从阿里云 DashScope API 迁移到 DeepSeek V4 Flash 模型。

## 配置步骤

1. **获取 DeepSeek API Key**
   - 访问 [DeepSeek 开放平台](https://platform.deepseek.com/)
   - 注册账号并创建 API Key

2. **修改 .env 文件**
   - 打开项目根目录下的 `.env` 文件
   - 将 `DEEPSEEK_API_KEY=your_deepseek_api_key_here` 中的 `your_deepseek_api_key_here` 替换为你的真实 API Key

## 主要修改文件

1. `.env` - 更新了 API Key 配置
2. `agents/base_agent.py` - 更新了：
   - API 基础 URL 为 `https://api.deepseek.com/v1`
   - 默认模型为 `deepseek-v4-flash`
   - 上下文窗口增加到 100 万 token
3. `app.py` - 更新了聊天接口的 API 配置和模型名称

## DeepSeek 模型说明

- 默认使用 `deepseek-v4-flash` 模型
- 同时也支持 `deepseek-v4-pro` 等其他模型
- DeepSeek V4 系列支持百万 token 超长上下文
- DeepSeek API 兼容 OpenAI 格式
- 完整模型列表请参考 DeepSeek 官方文档

## 运行测试

配置完成后，可以启动应用进行测试：

```bash
cd Agent
python app.py
```
