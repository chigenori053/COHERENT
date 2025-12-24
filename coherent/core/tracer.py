
from .state import State
from .action import Action

class Tracer:
    def start_episode(self, expression: str):
        pass

    def log_step(self, state: State, action: Action, result: dict):
        pass
