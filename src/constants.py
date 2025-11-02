from enum import auto, StrEnum


CMAP = {
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blud": 34,
    "purple": 35,
    "cyan": 36,
}


class Targets(StrEnum):
    IR = auto()
    C = auto()


class Ops(StrEnum):
    LOAD = auto()
    MUL = auto()
    ACCUMUL = auto()


class Commands(StrEnum):
    FOR = auto()


def color(text: str, c: str):
    return f"\033[{CMAP[c]}m{text}\033[0m"

class IR(StrEnum):
    LOAD = color("load", "cyan")
    ACCUMUL = color("accumul", "red")
    MUL = color("mul", "yellow")
    ADD = color("add", "yellow")
    FOR = color("for", "purple")
    IN = color("in", "purple")
