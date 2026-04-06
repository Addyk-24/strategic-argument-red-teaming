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

# 1. Import your actual environment
from environment import DebateEnvironment 
from schema.schemas import DebateAction

# Initialize FastAPI and your Environment
app = FastAPI()
env = DebateEnvironment()

# ==========================================
# PART 1: YOUR FASTAPI ENDPOINTS (For the Auto-Grader)
# ==========================================

@app.post("/reset")
def reset_api(topic: str = "Universal Basic Income is necessary"):
    # The hackathon bot will hit this!
    obs = env.reset(topic)
    return obs

@app.post("/step")
def step_api(action: DebateAction):
    # The hackathon bot will hit this!
    obs = env.step(action)
    return obs

# (Keep any other FastAPI routes you already had here)


# ==========================================
# PART 2: YOUR GRADIO UI (For the Human Judges)
# ==========================================

def reset_env_ui(topic, difficulty):
    """Wrapper for the UI to call env.reset()"""
    obs = env.reset(topic)
    chat_history = [{"role": "assistant", "content": obs.opponent_challenge}]
    return [chat_history, obs.phase, str(obs.reward), obs.done]

def step_env_ui(argument, phase_tag, chat_history):
    """Wrapper for the UI to call env.step()"""
    if not argument.strip():
        return [chat_history, gr.update(), gr.update(), gr.update(), gr.update()]

    action = DebateAction(argument=argument, phase_tag=phase_tag)
    obs = env.step(action)
    
    chat_history.append({"role": "user", "content": f"**[{phase_tag}]** {argument}"})
    chat_history.append({"role": "assistant", "content": obs.opponent_challenge})
    
    return [chat_history, obs.phase, str(obs.reward), obs.done, ""]

# Define your Gradio Blocks exactly as you built them
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ Strategic Argument Red-Teaming Simulator")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Environment Setup")
            topic_input = gr.Textbox(label="Debate Topic", value="Universal Basic Income is necessary")
            difficulty_input = gr.Dropdown(label="Difficulty/Task Level", choices=["Easy", "Medium", "Hard"], value="Medium")
            reset_btn = gr.Button("Reset Environment", variant="primary")
            
            gr.Markdown("### 📊 Observation Panel")
            current_phase_output = gr.Textbox(label="Current Phase", interactive=False)
            reward_output = gr.Textbox(label="Current Step Reward", interactive=False)
            is_done_output = gr.Checkbox(label="Is Done? (Episode Terminated)", interactive=False)
            
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat Interface (The Debate)")
            chatbot = gr.Chatbot(label="Debate History", height=450)
            
            gr.Markdown("### ⚡ Execute Action")
            with gr.Row():
                argument_input = gr.Textbox(label="Argument", placeholder="Type your compelling argument here...", scale=3)
                phase_tag_input = gr.Dropdown(label="Current Phase Tag", choices=["OPENING", "CHALLENGE", "REBUTTAL", "CONSOLIDATION", "CLOSING"], value="OPENING", scale=1)
            submit_btn = gr.Button("Submit Action", variant="primary")

    # Event Listeners
    reset_btn.click(fn=reset_env_ui, inputs=[topic_input, difficulty_input], outputs=[chatbot, current_phase_output, reward_output, is_done_output])
    submit_btn.click(fn=step_env_ui, inputs=[argument_input, phase_tag_input, chatbot], outputs=[chatbot, current_phase_output, reward_output, is_done_output, argument_input])


# ==========================================
# PART 3: MOUNT GRADIO ONTO FASTAPI
# ==========================================

# This is the magic line! It tells FastAPI to serve the Gradio UI at the root path ("/")
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)