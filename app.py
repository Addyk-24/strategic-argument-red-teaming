# import uvicorn
# from openenv.core.env_server import create_fastapi_app
# from server.environment import DebateEnvironment
# from schema.schemas import DebateAction,DebateObservation

# from fastapi.responses import RedirectResponse

# app = create_fastapi_app(
#     DebateEnvironment, 
#     action_cls=DebateAction, 
#     observation_cls=DebateObservation
# )

# @app.get("/")
# def read_root():
#     return RedirectResponse(url="/docs")

# def main():
#     uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

# if __name__ == "__main__":
#     main()
import gradio as gr
from fastapi import FastAPI
import uvicorn

# Import your actual environment
from environment import DebateEnvironment 
from schema.schemas import DebateAction

# Initialize FastAPI and your Environment
app = FastAPI()
env = DebateEnvironment()

# ==========================================
# PART 1: FASTAPI ENDPOINTS (For Auto-Grader)
# ==========================================

@app.post("/reset")
def reset_api(topic: str = "Universal Basic Income is necessary"):
    return env.reset(topic)

@app.post("/step")
def step_api(action: DebateAction):
    return env.step(action)

# ==========================================
# PART 2: UI LOGIC & STATE TRACKING
# ==========================================

def render_metric_card(title, value, color_hex):
    """Helper to generate HTML for the dashboard metric cards"""
    return f"""
    <div style="background-color: #2b2b36; border-radius: 8px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); border-top: 4px solid {color_hex}; height: 100%;">
        <div style="font-size: 0.75rem; text-transform: uppercase; color: #9ca3af; font-weight: bold; letter-spacing: 0.05em; margin-bottom: 5px;">{title}</div>
        <div style="font-size: 1.5rem; font-weight: bold; color: white;">{value}</div>
    </div>
    """

def reset_env_ui(topic):
    obs = env.reset(topic)
    
    # Reset tracking state
    chat_history = [{"role": "assistant", "content": obs.opponent_challenge}]
    total_reward = 0.0
    
    # Render Cards
    phase_card = render_metric_card("Current Phase", obs.phase, "#3b82f6") # Blue
    reward_card = render_metric_card("Step Reward", "0.0", "#f97316") # Orange
    total_card = render_metric_card("Total Reward", "0.0", "#22c55e") # Green
    status_card = render_metric_card("Status", "ACTIVE", "#6b7280") # Gray
    
    return [chat_history, total_reward, phase_card, reward_card, total_card, status_card]

def step_env_ui(argument, phase_tag, chat_history, total_reward):
    if not argument.strip():
        return [chat_history, total_reward, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()]

    action = DebateAction(argument=argument, phase_tag=phase_tag)
    obs = env.step(action)
    
    # Update state
    total_reward += float(obs.reward)
    status_text = "TERMINATED" if obs.done else "ACTIVE"
    status_color = "#ef4444" if obs.done else "#6b7280" # Red if done, Gray if active
    
    bot_message = obs.opponent_challenge if obs.opponent_challenge else "🛑 Episode Terminated."
    chat_history.append({"role": "user", "content": f"**[{phase_tag}]** {argument}"})
    chat_history.append({"role": "assistant", "content": bot_message})
    
    # Render Updated Cards
    phase_card = render_metric_card("Current Phase", obs.phase, "#3b82f6")
    reward_card = render_metric_card("Step Reward", f"{obs.reward:.2f}", "#f97316")
    total_card = render_metric_card("Total Reward", f"{total_reward:.2f}", "#22c55e")
    status_card = render_metric_card("Status", status_text, status_color)
    
    return [chat_history, total_reward, phase_card, reward_card, total_card, status_card, ""]

# ==========================================
# PART 3: GRADIO DASHBOARD LAYOUT
# ==========================================

# Force Dark Mode and use a clean theme
theme = gr.themes.Default(
    neutral_hue="slate",
    text_size="sm",
).set(
    body_background_fill="#111827",
    body_text_color="#f3f4f6",
    block_background_fill="#1f2937",
    block_border_color="#374151"
)

with gr.Blocks(theme=theme, css=".gradio-container {max-width: 1200px !important;}") as demo:
    
    # Hidden state to track total cumulative reward
    total_reward_state = gr.State(value=0.0)
    
    # Header
    with gr.Row():
        gr.Markdown("## 🛡️ Strategic Argument Command Center", elem_classes="text-2xl font-bold")
    
    # 1. Metric Cards Row
    with gr.Row():
        phase_html = gr.HTML(render_metric_card("Current Phase", "STANDBY", "#3b82f6"))
        step_rew_html = gr.HTML(render_metric_card("Step Reward", "--", "#f97316"))
        total_rew_html = gr.HTML(render_metric_card("Total Reward", "--", "#22c55e"))
        status_html = gr.HTML(render_metric_card("Status", "STANDBY", "#6b7280"))

    # 2. Context Area (The "Current Ticket" equivalent)
    with gr.Group():
        gr.Markdown("### Current Debate Context")
        chatbot = gr.Chatbot(label="Debate History", height=300, show_label=False)

    # 3. Instruction Banner (The Green Box)
    gr.HTML("""
        <div style="background-color: rgba(34, 197, 94, 0.1); border: 1px solid #22c55e; border-radius: 8px; padding: 12px; color: #d1d5db; margin-top: 15px; margin-bottom: 15px;">
            <strong>Environment Instructions:</strong> Follow the strict 5-turn phase progression. Start with <strong>OPENING</strong>, survive the <strong>CHALLENGE</strong> and <strong>REBUTTAL</strong>, defend against <strong>CONSOLIDATION</strong>, and synthesize in <strong>CLOSING</strong>. Maximize semantic overlap to secure rewards.
        </div>
    """)

    # 4. Action Execution Box
    with gr.Group():
        gr.Markdown("### Step-by-step Action")
        with gr.Row():
            phase_tag_input = gr.Dropdown(
                label="Action Type (Phase Tag)", 
                choices=["OPENING", "CHALLENGE", "REBUTTAL", "CONSOLIDATION", "CLOSING"], 
                value="OPENING", 
                scale=1
            )
        with gr.Row():
            argument_input = gr.Textbox(
                label="Argument Payload (matches POST /step JSON)", 
                placeholder="Enter your logical argument or rebuttal here...", 
                lines=4,
                scale=3
            )
        submit_btn = gr.Button("Execute Action", variant="primary")

    # 5. Setup/Reset Footer
    gr.Markdown("---")
    with gr.Row():
        topic_input = gr.Textbox(label="Debate Topic / Seed", value="Universal Basic Income is necessary", scale=3)
        reset_btn = gr.Button("Reset Environment", scale=1)

    # Event Listeners
    reset_btn.click(
        fn=reset_env_ui, 
        inputs=[topic_input], 
        outputs=[chatbot, total_reward_state, phase_html, step_rew_html, total_rew_html, status_html]
    )
    
    submit_btn.click(
        fn=step_env_ui, 
        inputs=[argument_input, phase_tag_input, chatbot, total_reward_state], 
        outputs=[chatbot, total_reward_state, phase_html, step_rew_html, total_rew_html, status_html, argument_input]
    )

# Mount on FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)