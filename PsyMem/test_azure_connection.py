 #test_azure_connection.py
from openai import OpenAI

# ---------- 替换为你的参数（与curl一致） ----------
AZURE_ENDPOINT = ""  # 必须以/结尾
API_KEY = ""  # 你的实际密钥
API_VERSION = ""  # 与curl中一致
DEPLOYMENT_NAME = ""  # 部署名称（与curl中一致）

# ---------- 构造与curl一致的base_url ----------
base_url = f"{AZURE_ENDPOINT}openai/deployments/{DEPLOYMENT_NAME}"

# ---------- 初始化客户端 ----------
client = OpenAI(
    api_key=API_KEY,
    base_url=base_url,
    default_headers={"api-version": API_VERSION}
)

# ---------- 发送与curl一致的测试请求 ----------
try:
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,  # 必须与部署名称一致
        messages=[{"role": "user", "content": "你好"}]
    )
    print("✅ Python 测试成功！模型响应：")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"❌ Python 测试失败：{str(e)}")