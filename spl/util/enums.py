from typing import Type, TypeVar
from enum import Enum

E = TypeVar("E", bound=Enum)

def str_to_enum(enum_cls: Type[E], value: str) -> E:
    try:
        return enum_cls[value]
    except KeyError:
        raise ValueError(f"Invalid value '{value}' for enum {enum_cls.__name__}")
