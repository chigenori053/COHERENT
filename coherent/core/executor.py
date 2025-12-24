
from .action import Action
from .state import State

class ActionExecutor:
    def __init__(self, runtime=None):
        self.runtime = runtime

    def execute(self, action: Action, state: State) -> dict:
        return {"valid": False, "error": "Executor stubbed", "before": state.current_expression, "after": state.current_expression}
