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

def render_metric_card(title, value, border_color):
    """Generates the HTML for the metric cards at the top of the command center."""
    return f"""
    <div style="border: 2px solid {border_color}; border-radius: 8px; padding: 12px; text-align: center; background-color: #0f172a;">
        <p style="margin: 0; font-size: 0.75em; color: #94a3b8; font-weight: 700; letter-spacing: 0.05em;">{title}</p>
        <h3 style="margin: 6px 0 0 0; color: #f8fafc; font-size: 1.25em;">{value}</h3>
    </div>
    """

def reset_env_ui(topic, difficulty):
    obs = env.reset(topic)
    
    # Modern Gradio format
    chat_history = [{"role": "assistant", "content": obs.opponent_challenge}]
    status_text = "Ended" if obs.done else "Active"
    
    return [
        render_metric_card("LAST STEP REWARD", "0.0", "#3b82f6"),
        render_metric_card("TOTAL REWARD", "0.0", "#f97316"),
        render_metric_card("CURRENT PHASE", obs.phase, "#22c55e"),
        render_metric_card("STATUS", status_text, "#64748b"),
        obs.opponent_challenge,  # Latest Opponent Challenge text
        chat_history,            # Chatbot update
        0.0                      # Reset total_reward state
    ]

def step_env_ui(argument, phase_tag, chat_history, total_reward):
    if not argument.strip():
        # Prevent empty submissions
        return [gr.update() for _ in range(8)]

    action = DebateAction(argument=argument, phase_tag=phase_tag)
    obs = env.step(action)
    
    # Update total reward
    new_total = total_reward + float(obs.reward)
    
    # Append to chat history
    bot_message = obs.opponent_challenge if obs.opponent_challenge else "🛑 The debate has concluded. Episode terminated."
    chat_history.append({"role": "user", "content": f"**[{phase_tag}]** {argument}"})
    chat_history.append({"role": "assistant", "content": bot_message})
    
    status_text = "Ended" if obs.done else "Active"
    status_color = "#ef4444" if status_text == "Ended" else "#64748b"
    
    return [
        render_metric_card("LAST STEP REWARD", f"{obs.reward:.4f}", "#3b82f6"),
        render_metric_card("TOTAL REWARD", f"{new_total:.4f}", "#f97316"),
        render_metric_card("CURRENT PHASE", obs.phase, "#22c55e"),
        render_metric_card("STATUS", status_text, status_color),
        bot_message,     # Latest Opponent Challenge text
        chat_history,    # Chatbot update
        new_total,       # Update total_reward state
        ""               # Clear argument payload textbox
    ]


# ==========================================
# PART 3: UI STYLING & LAYOUT
# ==========================================

custom_css = """
body { background-color: #1e293b; color: #f8fafc; }
.command-center { max-width: 950px !important; margin: 0 auto !important; }
.toast-standby { position: absolute; top: 15px; right: 20px; background-color: #475569; color: #2dd4bf; padding: 6px 16px; border-radius: 6px; font-weight: bold; z-index: 1000; font-size: 0.9em; border: 1px solid #0f172a; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4); }
.message-wrap .message.user { background-color: #0d9488 !important; border-color: #0d9488 !important; color: #ffffff !important; }
.message-wrap .message.bot { background-color: #334155 !important; border-color: #334155 !important; color: #f8fafc !important; }
.bold-value textarea, .bold-value input { font-weight: bold !important; color: #f8fafc !important; }
"""

corporate_dark_theme = gr.themes.Default(
    primary_hue="teal", secondary_hue="slate", neutral_hue="slate",
).set(
    body_background_fill="#1e293b", body_text_color="#f8fafc",
    background_fill_primary="#0f172a", background_fill_secondary="#1e293b",
    border_color_primary="#475569", input_background_fill="#334155",
    block_title_text_color="#94a3b8", button_primary_background_fill="#0d9488",
    button_primary_text_color="#ffffff"
)

with gr.Blocks(theme=corporate_dark_theme, css=custom_css) as demo:
    
    total_reward_state = gr.State(value=0.0)
    
    gr.HTML("<div class='toast-standby'>STANDBY</div>")
    
    with gr.Column(elem_classes="command-center"):
        
        # 1. Header Block
        gr.Markdown(
            "<h1 style='text-align: center; margin-bottom: 0;'>Strategic Argument Command Center</h1>"
            "<p style='text-align: center; color: #94a3b8; font-size: 1.1em; margin-top: 5px;'>"
            "Opening → Challenge → Rebuttal → Consolidation → Closing"
            "</p>"
        )
        
        # 2. Metric Cards Row
        with gr.Row():
            last_step_reward_card = gr.HTML(value=render_metric_card("LAST STEP REWARD", "0.0000", "#3b82f6")) 
            total_reward_card = gr.HTML(value=render_metric_card("TOTAL REWARD", "0.0000", "#f97316"))       
            current_phase_card = gr.HTML(value=render_metric_card("CURRENT PHASE", "STANDBY", "#22c55e")) 
            status_card = gr.HTML(value=render_metric_card("STATUS", "Standby", "#64748b"))                   
            
        # 3. Current Opponent State
        with gr.Group():
            gr.Markdown("### Latest Opponent Challenge")
            latest_challenge_output = gr.Textbox(
                show_label=False, interactive=False, lines=4,
                value="Initialize the environment to begin the debate.",
                elem_classes="bold-value"
            )
            
        # 4. Instruction Banner
        gr.HTML("""
        <div style="background-color: #064e3b; color: #34d399; padding: 12px; border-radius: 8px; border: 1px solid #047857; margin-top: 15px; margin-bottom: 15px;">
            <strong>Environment Constraints:</strong> Determine the optimal phase tag for your argument to maximize reward against the language model. Move from OPENING, through CHALLENGE and REBUTTAL, to CLOSING successfully.
        </div>
        """)
        
        # 5. Step-by-Step Action Container
        with gr.Group():
            gr.Markdown("### Execute Action")
            with gr.Row():
                phase_tag_input = gr.Dropdown(
                    label="Action Type (Phase Tag)", 
                    choices=["OPENING", "CHALLENGE", "REBUTTAL", "CONSOLIDATION", "CLOSING"], 
                    value="OPENING"
                )
            with gr.Row():
                argument_input = gr.Textbox(
                    label="Argument Payload", 
                    placeholder="Type your compelling argument here...", 
                    lines=3
                )
            with gr.Row():
                topic_input = gr.Dropdown(
                    label="Debate Topic", 
                    choices=["Universal Basic Income is necessary", "AI Regulation is required", "Space Exploration is vital"], 
                    value="Universal Basic Income is necessary"
                )
                difficulty_input = gr.Dropdown(label="Difficulty", choices=["Easy", "Medium", "Hard"], value="Medium")
                
            with gr.Row():
                submit_btn = gr.Button("EXECUTE ACTION", variant="primary")
                reset_btn = gr.Button("RESET ENVIRONMENT", variant="secondary")

        # 6. Debate Timeline
        with gr.Group():
            gr.Markdown("### Action Timeline (Debate History)")
            chatbot = gr.Chatbot(
                show_label=False,
                height=400,
                elem_classes="message-value",
            )

    # --- Event Listeners ---
    reset_btn.click(
        fn=reset_env_ui,
        inputs=[topic_input, difficulty_input],
        outputs=[last_step_reward_card, total_reward_card, current_phase_card, status_card, latest_challenge_output, chatbot, total_reward_state]
    )
    
    submit_btn.click(
        fn=step_env_ui,
        inputs=[argument_input, phase_tag_input, chatbot, total_reward_state],
        outputs=[last_step_reward_card, total_reward_card, current_phase_card, status_card, latest_challenge_output, chatbot, total_reward_state, argument_input]
    )


# ==========================================
# PART 4: MOUNT TO FASTAPI
# ==========================================
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)