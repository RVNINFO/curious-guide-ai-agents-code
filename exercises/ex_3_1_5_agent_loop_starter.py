"""Exercise starter (end of Part 3): implement the agent loop from Section 3.1.5.
Your solution should pass tests/test_exercises.py (swap the import to this module).
A reference solution is in solutions/ex_3_1_5_agent_loop.py - try first!
"""
from dataclasses import dataclass, field


@dataclass
class Call:
    id: str
    name: str
    arguments: dict


@dataclass
class Response:
    content: str = ""
    tool_calls: list = field(default_factory=list)


class FakeModel:
    """Plays back a script of responses. Once the script ends, it repeats the last one."""
    def __init__(self, script):
        self.script, self.i = script, 0

    def chat(self, messages, tools):
        r = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return r


class Tool:
    def __init__(self, name, fn):
        self.name, self.fn, self.schema = name, fn, {"name": name}

    def run(self, **kwargs):
        return self.fn(**kwargs)


def run_agent(model, user_input, tools, max_steps=5):
    """TODO: implement the loop from Section 3.1.5.
    Return the model's final text, or the step-limit message."""
    raise NotImplementedError
