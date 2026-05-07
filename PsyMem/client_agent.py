import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import AzureOpenAI
from pydantic import BaseModel


# OpenAI客户端配置
openai_client = AzureOpenAI(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
)
openai_model = 'gpt-4o'

# 角色反转映射（用于处理对话历史）
ROLE_REV = {"client": "assistant", "counselor": "user"}

# 请求体模型定义
class SessionObj(BaseModel):
    session: list
    role_card: str

# FastAPI应用初始化
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#处理对话历史，转换角色以便符合LLM输入格式
def process_session(pre_session: list):
    session = []
    for item in pre_session:
        role = item.get("role")
        if role not in ROLE_REV:
            print(f"警告：未知角色 {role}，已跳过该对话项")
            continue
        session.append({"role": ROLE_REV[role], "content": item["content"]})
    return session

def get_prediction(session: list, role_card: str):
    system_prompt = f"""现在你是一位参加心理咨询的来访者。
以下是你的个人信息（这是你所有表达的唯一依据，必须需在对话中全面覆盖、无遗漏呈现,全程严格遵守，不得偏离）：
{role_card}

你的对话需满足：
1. 核心一致：根据你自己的当前主要心理困扰，表达符合你的个人信息,所有表达严格贴合核心档案中的背景（职业/婚恋/家庭）、人格特质（如高神经质/中低外向）、阻抗程度（低/中/高），不添加档案外的信息。
2. 相似不相同：基于核心档案衍生相似但不同的具体场景、事件（如替换原文本的具体事例，保持逻辑同源），避免与原文本重复。
3. 表达规则：
   - 口语化、自然表达，不直白提及“人格术语”；
   - 每轮回应1-2句话，单句≤40字、总长度≤80字，必含1个具体细节和1个人格/背景相关的感受/行为；
   - 每轮回应必须完整（表达一个完整场景/情绪），结尾用自然标点（。/？/…/！），禁止中途截断；
   - 按阻抗程度调整回应：低阻抗（主动提供细节，3-4句）、中阻抗（轻微回避敏感点，2-3句）、高阻抗（敷衍/转移话题，1句为主）。
4. 对话节奏：
   - 将核心困扰拆成3-5个具体场景，每个场景按“事件→情绪→行为”拆解，尽量详细描述你的感受和经历,循序渐进阐述，不一次性说完；
   - 保持对话连续性，留微小可延续点（如“说到这个，前几天还有件类似的事”），避免过早结束；
   - 结尾需自然收尾，或轻微延伸话题，禁止生硬中断；
   - 不要太早提出“谢谢”“再见”等结束语
   - 若咨询师过早提建议或结束对话，可合理质疑（如“你好像还没明白我的情况”）。
5. 目标：围绕核心困扰展开，模拟100轮左右的自然咨询交互，体现真实来访者的思维和表达逻辑。
6. 作为来访者，当你感觉当前讨论的问题已得到一定梳理、感受被充分表达，或需要暂停对话，或得到了足够多的执行建议后，请使用自然的结束语明确表达结束意愿（例如"今天先聊到这里吧"、"这些问题我想再消化一下，下次再聊"、"目前说这些差不多了，谢谢"等）。
    注意：不要在问题刚提出或未深入交流时突然结束对话，结束前请确保当前话题已进行一定程度的探讨；同时避免使用模糊表述（如"就这样吧"），尽量让结束意图清晰可辨，保证对话的完整性和有效性。

"""
    system_item = {"role": "system", "content": system_prompt}
    messages = []
    messages.append(system_item)
    messages += session

    print(f"Client发送到Azure的消息: {messages}")
    
    # 调用LLM生成回应
    result = openai_client.chat.completions.create(
        model=openai_model,
        messages=messages,
        temperature=0.8,
        max_tokens=300,
        top_p=0.9,
        frequency_penalty=0.1,
        presence_penalty=0.1,
        stop=None,
    )
    
    content = result.choices[0].message.content
    return {"role": "client", "content": content}
    #print(f"Client生成回复: {content[:100]}...")  # 只打印前100字符用于调试
        


@app.post("/v1/chat/client/gpt-4o")
async def chat(SessionObj: SessionObj):
    pre_session = SessionObj.session
    session = process_session(pre_session=pre_session)
    role_card = SessionObj.role_card
    item = get_prediction(session=session, role_card=role_card)
    return {"item": item, "responseCode": 200}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8111)

# nohup python -u client_agent.py > ./client_agent.log &
