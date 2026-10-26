"""Tiny framework-free types shared by Listing 3.1.5 and exercise 3.1.5."""
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

    @property
    def message(self):  # the loop appends response.message to the history
        return self


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
