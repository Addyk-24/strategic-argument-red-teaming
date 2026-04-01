import uvicorn
from openenv.core.env_server import create_fastapi_app
from server.environment import DebateEnvironment
from schema.schemas import DebateAction,DebateObservation

app = create_fastapi_app(
    DebateEnvironment, 
    action_cls=DebateAction, 
    observation_cls=DebateObservation
)

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()