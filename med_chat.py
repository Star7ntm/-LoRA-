import gradio as gr
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="0000",
)


def normalize_message_content(content):
    """规范化消息内容，确保返回字符串"""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(str(item["text"]))
            elif isinstance(item, str):
                text_parts.append(item)
        return " ".join(text_parts) if text_parts else ""
    else:
        return str(content) if content else ""

class StreamChatBot:
    def __init__(self, model: str = "qwen3"):
        self.model = model

    async def stream_response(self, message: str, history: list, medical_info: dict):
        messages = []
        system_prompt = "你是由达星辰，陈欣悦，彭延浩三人共同训练的医疗AI助手，基于Qwen3-1.7B模型进行微调训练。你专注于提供医疗健康相关的咨询和建议，特别是中医诊疗和健康咨询。请用中文回答用户的问题。"
        messages.append({"role": "system", "content": system_prompt})

        for msg in history:
            if isinstance(msg, dict):
                role = msg.get("role")
                content = normalize_message_content(msg.get("content"))
            elif isinstance(msg, (list, tuple)) and len(msg) == 2:
                messages.append({"role": "user", "content": normalize_message_content(msg[0])})
                messages.append({"role": "assistant", "content": normalize_message_content(msg[1])})
                continue
            else:
                continue
            
            if role in ["user", "assistant"] and content:
                messages.append({"role": role, "content": content})

        user_message = "【患者医疗信息】\n"
        if medical_info.get("department"):
            user_message += f"就诊科室：{medical_info['department']}\n"
        if medical_info.get("symptoms"):
            symptoms_list = medical_info["symptoms"] if isinstance(medical_info["symptoms"], list) else []
            if symptoms_list:
                user_message += f"主要症状：{', '.join(symptoms_list)}\n"
        if medical_info.get("present_illness"):
            user_message += f"现病史：{medical_info['present_illness']}\n"
        if medical_info.get("past_history"):
            user_message += f"既往史：{medical_info['past_history']}\n"
        if medical_info.get("current_symptoms"):
            user_message += f"刻下症：{medical_info['current_symptoms']}\n"
        if medical_info.get("allergy_history"):
            user_message += f"过敏史：{medical_info['allergy_history']}\n"
        if medical_info.get("tcm_diagnosis"):
            user_message += f"中医四诊：{medical_info['tcm_diagnosis']}\n"
        if medical_info.get("physical_exam"):
            user_message += f"体格检查：{medical_info['physical_exam']}\n"
        if medical_info.get("diagnosis_name"):
            user_message += f"诊断名称：{medical_info['diagnosis_name']}\n"
        if medical_info.get("tcm_syndrome"):
            user_message += f"中医症候：{medical_info['tcm_syndrome']}\n"

        user_message += f"\n【用户问题】\n{message}"
        user_message += "\n\n请基于以上患者信息提供专业的中医诊疗建议。"
        messages.append({"role": "user", "content": user_message})
        
        try:
            stream = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                max_tokens=4096,
                temperature=0.7,
            )
            async for chunk in stream:
                if (chunk.choices and chunk.choices[0].delta.content is not None):
                    content = chunk.choices[0].delta.content
                    yield content
        except Exception as e:
            yield f"抱歉，发生了错误: {str(e)}"

chat_bot = StreamChatBot()

async def predict(message, history, medical_info):
    full_response = ""
    async for content in chat_bot.stream_response(message, history, medical_info):
        cleaned_content = content.replace("<think>", "").replace("</think>", "")
        full_response += cleaned_content
        yield full_response

# Apple官网风格 - 侧边栏折叠设计
APPLE_STYLE_CSS = """
:root {
    --apple-blue: #0071E3;
    --apple-white: #FFFFFF;
    --apple-black: #1D1D1F;
    --apple-gray-light: #F5F5F7;
    --apple-gray-medium: #86868B;
    --apple-gray-dark: #6E6E73;
    --apple-border: #D2D2D7;
    --apple-shadow: rgba(0, 0, 0, 0.1);
}

* {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", Arial, sans-serif !important;
}

body {
    background: var(--apple-white) !important;
    margin: 0;
    padding: 0;
}

.gradio-container {
    background: var(--apple-white) !important;
    padding: 0 !important;
    max-width: 100% !important;
}

/* Hero区域 */
.hero-section {
    background: var(--apple-white);
    padding: 30px 20px 20px;
    text-align: center;
    border-bottom: 1px solid var(--apple-border);
}

h1 {
    font-size: 48px;
    font-weight: 600;
    letter-spacing: -1px;
    color: var(--apple-black);
    margin: 0 0 8px 0;
}

.hero-subtitle {
    font-size: 19px;
    color: var(--apple-gray-medium);
    margin: 0;
}

/* 侧边栏 */
#medical-sidebar-column {
    position: fixed !important;
    left: -420px !important;
    top: 0 !important;
    width: 420px !important;
    height: 100vh !important;
    background: var(--apple-white) !important;
    border-right: 1px solid var(--apple-border) !important;
    box-shadow: 2px 0 20px rgba(0, 0, 0, 0.15) !important;
    transition: left 0.35s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
    z-index: 1000 !important;
    overflow-y: auto !important;
    padding: 30px 24px !important;
    box-sizing: border-box !important;
    display: block !important;
}

#medical-sidebar-column.open {
    left: 0 !important;
}

/* 确保侧边栏内容可见 */
#medical-sidebar-column .gr-form,
#medical-sidebar-column .gr-form-group,
#medical-sidebar-column .gr-dropdown,
#medical-sidebar-column .gr-checkboxgroup,
#medical-sidebar-column .gr-textbox {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}

.sidebar {
    position: fixed !important;
    left: -420px !important;
    top: 0 !important;
    width: 420px !important;
    height: 100vh !important;
    background: var(--apple-white) !important;
    border-right: 1px solid var(--apple-border) !important;
    box-shadow: 2px 0 20px rgba(0, 0, 0, 0.15) !important;
    transition: left 0.35s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
    z-index: 1000 !important;
    overflow-y: auto !important;
}

.sidebar.open {
    left: 0 !important;
}

.sidebar-toggle {
    position: fixed !important;
    left: 20px !important;
    top: 20px !important;
    z-index: 1001 !important;
    background: #1D1D1F !important;
    color: #FFFFFF !important;
    border: 3px solid #FFFFFF !important;
    border-radius: 14px !important;
    padding: 16px 32px !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    cursor: pointer !important;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4), 0 0 0 2px rgba(255, 255, 255, 0.3) !important;
    transition: all 0.3s ease !important;
    min-width: 130px !important;
    text-align: center !important;
    letter-spacing: 0.5px !important;
    text-shadow: none !important;
}

.sidebar-toggle:hover {
    background: #6E6E73 !important;
    transform: translateY(-3px) scale(1.05) !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5), 0 0 0 2px rgba(255, 255, 255, 0.5) !important;
}

.sidebar-toggle:active {
    transform: translateY(-1px) scale(1.02) !important;
}

.sidebar-overlay {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100% !important;
    height: 100vh !important;
    background: rgba(0, 0, 0, 0.3) !important;
    z-index: 999 !important;
    opacity: 0 !important;
    pointer-events: none !important;
    transition: opacity 0.3s ease !important;
}

.sidebar.open ~ .sidebar-overlay,
#medical-sidebar-column.open ~ .sidebar-overlay {
    opacity: 1 !important;
    pointer-events: auto !important;
}

/* 遮罩层显示控制 */
.sidebar-overlay.open {
    opacity: 1 !important;
    pointer-events: auto !important;
}

/* 主内容区域 */
.chat-main-area {
    width: 100% !important;
    max-width: 1200px !important;
    padding: 20px 40px 40px !important;
    margin: 0 auto !important;
}

/* 按钮 */
button.primary, .gr-button-primary {
    background: var(--apple-black) !important;
    color: var(--apple-white) !important;
    border: none !important;
    border-radius: 22px !important;
    padding: 14px 28px !important;
    font-size: 17px !important;
    font-weight: 400 !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    min-height: 44px !important;
}

button.primary:hover, .gr-button-primary:hover {
    background: var(--apple-gray-dark) !important;
    transform: scale(1.02) !important;
}

button.secondary, .gr-button-secondary {
    background: var(--apple-white) !important;
    color: var(--apple-black) !important;
    border: 1px solid var(--apple-border) !important;
    border-radius: 22px !important;
    padding: 14px 28px !important;
    font-size: 17px !important;
    font-weight: 400 !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    min-height: 44px !important;
}

button.secondary:hover, .gr-button-secondary:hover {
    background: var(--apple-gray-light) !important;
    transform: scale(1.02) !important;
}

/* 输入框 */
input[type="text"], textarea, select {
    border: 1px solid var(--apple-border) !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    font-size: 17px !important;
    background: var(--apple-white) !important;
    color: var(--apple-black) !important;
    transition: all 0.3s ease !important;
}

input[type="text"]:focus, textarea:focus, select:focus {
    outline: none !important;
    border-color: var(--apple-black) !important;
    box-shadow: 0 0 0 4px rgba(29, 29, 31, 0.1) !important;
}

/* 消息气泡 */
.message-user {
    background: var(--apple-black) !important;
    color: var(--apple-white) !important;
    border-radius: 20px 20px 4px 20px !important;
    padding: 14px 20px !important;
    margin: 12px 0 !important;
    max-width: 75% !important;
    margin-left: auto !important;
}

.message-assistant {
    background: var(--apple-gray-light) !important;
    color: var(--apple-black) !important;
    border-radius: 20px 20px 20px 4px !important;
    padding: 14px 20px !important;
    margin: 12px 0 !important;
    max-width: 75% !important;
    margin-right: auto !important;
}

label {
    font-size: 15px !important;
    font-weight: 500 !important;
    color: var(--apple-black) !important;
    margin-bottom: 10px !important;
}

.input-row {
    margin-top: 20px !important;
    gap: 12px !important;
}

.button-row {
    margin-top: 12px !important;
    justify-content: flex-end !important;
}
"""

def create_chat_interface():
    with gr.Blocks(css=APPLE_STYLE_CSS) as demo:
        # 侧边栏切换按钮
        gr.HTML("""
        <button class="sidebar-toggle" id="sidebar-toggle" onclick="toggleSidebar()">
            <span id="toggle-text">患者信息</span>
        </button>
        <script>
            function toggleSidebar() {
                // 尝试多种方式找到侧边栏元素
                let sidebar = document.getElementById('medical-sidebar-column');
                if (!sidebar) {
                    // 如果找不到，尝试通过class查找
                    sidebar = document.querySelector('.medical-sidebar');
                }
                if (!sidebar) {
                    // 最后尝试查找包含医疗信息的Column
                    const columns = document.querySelectorAll('.gr-column');
                    for (let col of columns) {
                        if (col.querySelector('h2') && col.querySelector('h2').textContent.includes('患者医疗信息')) {
                            sidebar = col;
                            sidebar.id = 'medical-sidebar-column';
                            break;
                        }
                    }
                }
                
                const overlay = document.getElementById('sidebar-overlay');
                const toggleText = document.getElementById('toggle-text');
                
                if (sidebar) {
                    sidebar.classList.toggle('open');
                    // 强制设置样式
                    if (sidebar.classList.contains('open')) {
                        sidebar.style.left = '0px';
                        sidebar.style.display = 'block';
                        sidebar.style.visibility = 'visible';
                        if (overlay) {
                            overlay.style.opacity = '1';
                            overlay.style.pointerEvents = 'auto';
                        }
                        if (toggleText) toggleText.textContent = '关闭';
                    } else {
                        sidebar.style.left = '-420px';
                        if (overlay) {
                            overlay.style.opacity = '0';
                            overlay.style.pointerEvents = 'none';
                        }
                        if (toggleText) toggleText.textContent = '患者信息';
                    }
                } else {
                    console.error('无法找到侧边栏元素');
                }
            }
            
            // 页面加载完成后确保侧边栏初始状态
            function initSidebar() {
                let sidebar = document.getElementById('medical-sidebar-column');
                if (!sidebar) {
                    const columns = document.querySelectorAll('.gr-column');
                    for (let col of columns) {
                        if (col.querySelector('h2') && col.querySelector('h2').textContent.includes('患者医疗信息')) {
                            sidebar = col;
                            sidebar.id = 'medical-sidebar-column';
                            sidebar.classList.add('medical-sidebar');
                            break;
                        }
                    }
                }
                if (sidebar) {
                    sidebar.style.position = 'fixed';
                    sidebar.style.left = '-420px';
                    sidebar.style.top = '0';
                    sidebar.style.width = '420px';
                    sidebar.style.height = '100vh';
                    sidebar.style.zIndex = '1000';
                    sidebar.style.display = 'block';
                    sidebar.style.visibility = 'visible';
                    sidebar.style.background = '#FFFFFF';
                    sidebar.style.borderRight = '1px solid #D2D2D7';
                    sidebar.style.boxShadow = '2px 0 20px rgba(0, 0, 0, 0.15)';
                    sidebar.style.transition = 'left 0.35s cubic-bezier(0.25, 0.46, 0.45, 0.94)';
                    sidebar.style.overflowY = 'auto';
                    sidebar.style.padding = '30px 24px';
                    sidebar.style.boxSizing = 'border-box';
                }
            }
            
            // 等待Gradio渲染完成
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', initSidebar);
            } else {
                initSidebar();
            }
            
            // 延迟执行，确保Gradio组件已渲染
            setTimeout(initSidebar, 1000);
        </script>
        """)
        
        # Hero区域
        gr.HTML("""
        <div class="hero-section">
            <h1>中医智能诊疗助手</h1>
            <p class="hero-subtitle">基于AI的专业中医诊疗咨询平台</p>
        </div>
        """)
        
        # 侧边栏 - 医疗信息（使用Column包装，然后用CSS定位）
        with gr.Column(visible=True, elem_id="medical-sidebar-column"):
            gr.HTML('<h2 style="margin-top: 0; margin-bottom: 28px; font-size: 28px;">患者医疗信息</h2>')
            
            department = gr.Dropdown(
                choices=["内科", "外科", "儿科", "妇科", "骨科", "神经科", "心血管科", 
                        "消化科", "呼吸科", "内分泌科", "皮肤科", "眼科", "耳鼻喉科",
                        "中医科", "康复科", "急诊科", "其他"],
                label="就诊科室",
                value="中医科",
                multiselect=False
            )
            
            symptoms = gr.CheckboxGroup(
                choices=["发热", "咳嗽", "头痛", "腹痛", "腹泻", "呕吐", "乏力", 
                        "失眠", "焦虑", "胸闷", "心悸", "关节痛", "皮疹", "咳嗽有痰",
                        "咽痛", "鼻塞", "流涕", "食欲不振", "恶心", "便秘", "尿频",
                        "腰酸", "背痛", "四肢无力", "头晕", "耳鸣", "视力模糊"],
                label="主要症状（可多选）",
                value=[]
            )
            
            present_illness = gr.Textbox(label="现病史", value="患者主诉咳嗽、咳痰3天，伴有发热", lines=2)
            past_history = gr.Textbox(label="既往史", value="无重大疾病史，无手术史", lines=2)
            current_symptoms = gr.Textbox(label="刻下症（详细描述）", value="咳嗽频作，痰黄粘稠，发热38.5℃，口渴，咽痛", lines=3)
            allergy_history = gr.Textbox(label="过敏史", value="无药物及食物过敏史", lines=2)
            tcm_diagnosis = gr.Textbox(label="中医四诊", value="舌红苔黄腻，脉浮数", lines=2)
            physical_exam = gr.Textbox(label="体格检查", value="咽部充血，扁桃体I度肿大，双肺呼吸音粗", lines=2)
            diagnosis_name = gr.Textbox(label="诊断名称", value="急性支气管炎", lines=2)
            tcm_syndrome = gr.Textbox(label="中医症候", value="风热犯肺证", lines=2)
        
        gr.HTML('<div class="sidebar-overlay" id="sidebar-overlay" onclick="toggleSidebar()"></div>')
        
        # 主内容区域 - 聊天窗口
        gr.HTML('<div class="chat-main-area">')
        
        chatbot = gr.Chatbot(label=None, height=650, show_label=False, container=True)
        
        with gr.Row(elem_classes="input-row"):
            msg = gr.Textbox(label=None, placeholder="请输入关于患者诊疗的问题...", scale=5, container=False, show_label=False)
            submit_btn = gr.Button("发送", variant="primary", scale=1, min_width=100)
        
        with gr.Row(elem_classes="button-row"):
            clear_btn = gr.Button("清空对话", variant="secondary")
        
        gr.HTML('</div>')
        
        medical_info = {
            "department": department,
            "symptoms": symptoms,
            "present_illness": present_illness,
            "past_history": past_history,
            "current_symptoms": current_symptoms,
            "allergy_history": allergy_history,
            "tcm_diagnosis": tcm_diagnosis,
            "physical_exam": physical_exam,
            "diagnosis_name": diagnosis_name,
            "tcm_syndrome": tcm_syndrome
        }
        
        def get_medical_info(*args):
            return {
                "department": args[0],
                "symptoms": args[1],
                "present_illness": args[2],
                "past_history": args[3],
                "current_symptoms": args[4],
                "allergy_history": args[5],
                "tcm_diagnosis": args[6],
                "physical_exam": args[7],
                "diagnosis_name": args[8],
                "tcm_syndrome": args[9]
            }
        
        async def respond(message, chat_history, *medical_args):
            if not message.strip():
                yield chat_history, ""
                return
            
            medical_info_dict = get_medical_info(*medical_args)
            if chat_history is None:
                chat_history = []
            
            normalized_history = []
            for item in chat_history:
                if isinstance(item, dict):
                    normalized_history.append(item)
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    normalized_history.append({"role": "user", "content": str(item[0])})
                    normalized_history.append({"role": "assistant", "content": str(item[1])})
            
            normalized_history.append({"role": "user", "content": message})
            normalized_history.append({"role": "assistant", "content": ""})
            
            full_response = ""
            async for content in predict(message, normalized_history[:-2], medical_info_dict):
                full_response = content
                normalized_history[-1] = {"role": "assistant", "content": full_response}
                yield normalized_history, ""
        
        submit_btn.click(respond, [msg, chatbot] + list(medical_info.values()), [chatbot, msg])
        msg.submit(respond, [msg, chatbot] + list(medical_info.values()), [chatbot, msg])
        clear_btn.click(lambda: [], None, chatbot)
    
    return demo

if __name__ == "__main__":
    import socket
    import sys
    
    def check_api_server():
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('127.0.0.1', 8000))
            sock.close()
            return result == 0
        except:
            return False
    
    if not check_api_server():
        print("=" * 70)
        print("警告: API 服务器未运行或无法连接")
        print("=" * 70)
        print("请先启动 API 服务器：")
        print("  python api_server.py --model-name ./models/Qwen/Qwen3-1.7B --lora-checkpoint ./sft_output_high_quality/checkpoint-3000")
        print("=" * 70)
        response = input("是否继续启动 WebUI？(y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    demo = create_chat_interface()
    print("正在启动 WebUI...")
    try:
        demo.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
            inbrowser=True,
            css=APPLE_STYLE_CSS,
            theme=gr.themes.Soft()
        )
    except Exception as e:
        print(f"启动失败: {e}")
        try:
            demo.launch(share=True, server_port=7860, show_error=True, inbrowser=True, css=APPLE_STYLE_CSS, theme=gr.themes.Soft())
        except Exception as e2:
            print(f"备用方案也失败: {e2}")
            raise

