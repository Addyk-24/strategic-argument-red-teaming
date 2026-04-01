import uvicorn
from openenv.core.env_server import create_fastapi_app
from server.environment import DebateEnvironment
from schema.schemas import DebateAction,DebateObservation

from fastapi.responses import RedirectResponse

app = create_fastapi_app(
    DebateEnvironment, 
    action_cls=DebateAction, 
    observation_cls=DebateObservation
)

@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()