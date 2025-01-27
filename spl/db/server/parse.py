# parse.py
import types
from enum import Enum
from inspect import signature
from typing import Any, Dict, get_type_hints, Union, get_origin, get_args

def convert_to_type(value, expected_type):
    """
    Convert `value` to the expected Python type/Enum/Union/etc.
    Handles:
      - Enum by name
      - bool from 'true'/'false'/'1'/'0'
      - list[...] by recursively converting each item
      - Union[...] by trying each variant
      - other types via direct constructor

    Raises ValueError if conversion fails.
    """
    if value is None:
        # Check if expected_type can accept None, i.e. Union[..., None]
        if get_origin(expected_type) is Union or isinstance(expected_type, types.UnionType):
            if type(None) in get_args(expected_type):
                return None
        raise ValueError(f"Cannot convert None to {expected_type}")

    # 1) If it's an Enum type, convert by name
    if isinstance(expected_type, type) and issubclass(expected_type, Enum):
        # Convert string -> Enum member
        try:
            return expected_type[value]
        except KeyError:
            raise ValueError(f"Cannot convert {value!r} to Enum {expected_type.__name__}")

    # 2) If it's a list type => convert each item
    if get_origin(expected_type) is list:
        (inner_type,) = get_args(expected_type)  # e.g. List[str]
        if not isinstance(value, list):
            raise ValueError(f"Expected list for {expected_type}, got {type(value).__name__}")
        return [convert_to_type(item, inner_type) for item in value]

    # 3) If it's a Union => try each sub-type
    if get_origin(expected_type) is Union or isinstance(expected_type, types.UnionType):
        for sub_type in get_args(expected_type):
            try:
                return convert_to_type(value, sub_type)
            except ValueError:
                pass
        raise ValueError(f"Cannot convert {value!r} to any of {get_args(expected_type)}")

    # 4) Special handling for bool from string
    if expected_type is bool and isinstance(value, str):
        lower_val = value.lower()
        if lower_val in ['true', '1']:
            return True
        if lower_val in ['false', '0']:
            return False
        raise ValueError(f"Cannot convert {value!r} to bool")

    # 5) Otherwise, try direct constructor
    try:
        return expected_type(value)
    except Exception as e:
        raise ValueError(f"Error converting {value!r} to {expected_type}: {e}")


def parse_args_with_types(func, raw_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the provided `raw_args` (a dict of JSON fields) against the signature
    of `func`, converting each param to its annotated type. If a required param
    is missing, raises ValueError. If any leftover fields exist in `raw_args`,
    also raises ValueError (to ensure strictness).

    Example usage within a route:
      parsed_data = parse_args_with_types(db_adapter_server.update_instance, data)

    Then call:
      result = await db_adapter_server.update_instance(**parsed_data)
    """
    sig = signature(func)
    hints = get_type_hints(func)
    parsed_args = {}

    # Gather declared parameters (excluding 'self' for instance methods).
    declared_params = {
        name: param
        for name, param in sig.parameters.items()
        if name != 'self'
    }

    # 1) For each declared param:
    for param_name, param in declared_params.items():
        # If param is missing from raw_args and there's no default => error
        if param.default is param.empty and param_name not in raw_args:
            raise ValueError(f"Missing required parameter: '{param_name}'")

        if param_name in raw_args:
            # If param has a type hint, convert
            if param_name in hints:
                expected_type = hints[param_name]
                parsed_args[param_name] = convert_to_type(raw_args[param_name], expected_type)
            else:
                # No hint => pass raw
                parsed_args[param_name] = raw_args[param_name]
        else:
            # param not in JSON => use default
            parsed_args[param_name] = param.default

    # 2) Disallow leftover JSON fields not in the function signature
    leftover = set(raw_args.keys()) - set(declared_params.keys())
    if leftover:
        # Could ignore them, but typically we want to fail fast
        leftover_list = ", ".join(sorted(leftover))
        raise ValueError(f"Unexpected extra field(s) in JSON: {leftover_list}")

    return parsed_args
