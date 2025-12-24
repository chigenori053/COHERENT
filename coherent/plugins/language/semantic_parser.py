
from .semantic_types import SemanticIR, TaskType, InputItem

class RuleBasedSemanticParser:
    def __init__(self):
        pass

    def parse(self, text: str) -> SemanticIR:
        # Stub implementation
        return SemanticIR(
            task=TaskType.SOLVE,
            inputs=[InputItem(value=text)]
        )
