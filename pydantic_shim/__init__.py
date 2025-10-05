"""Minimal Pydantic-compatible shim for environments without the dependency."""
from __future__ import annotations

from dataclasses import MISSING
from typing import Any, Callable, Dict, Optional


class ConfigDict(dict):
    """Lightweight stand-in for Pydantic's ConfigDict."""


class _FieldInfo:
    def __init__(self, default: Any = MISSING, default_factory: Callable[[], Any] | None = None):
        self.default = default
        self.default_factory = default_factory


def Field(default: Any = MISSING, default_factory: Callable[[], Any] | None = None) -> Any:
    return _FieldInfo(default, default_factory)


def field_validator(name: str, *, mode: str = "after") -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        setattr(func, "__field_name__", name)
        setattr(func, "__field_mode__", mode)
        return func

    return decorator


class BaseModel:
    model_config: ConfigDict = ConfigDict()

    def __init__(self, **data: Any) -> None:
        values = self._validate_dict(data if isinstance(data, dict) else data.__dict__)
        for key, value in values.items():
            object.__setattr__(self, key, value)

    @classmethod
    def _validate_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        annotations = getattr(cls, "__annotations__", {})
        field_values: Dict[str, Any] = {}
        validators = _collect_validators(cls)
        for field_name, annotation in annotations.items():
            info = getattr(cls, field_name, _FieldInfo())
            if isinstance(info, _FieldInfo):
                if field_name in data:
                    value = data[field_name]
                elif info.default is not MISSING:
                    value = info.default
                elif info.default_factory is not None:
                    value = info.default_factory()
                else:
                    raise ValueError(f"Missing value for field '{field_name}'")
            else:
                value = data.get(field_name, info)
            value = _apply_validators(validators, field_name, value, mode="before")
            value = _coerce(annotation, value)
            value = _apply_validators(validators, field_name, value, mode="after")
            field_values[field_name] = value
        return field_values

    @classmethod
    def model_validate(cls, data: Any) -> "BaseModel":
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            raise ValueError("Input for model validation must be a mapping.")
        values = cls._validate_dict(data)
        instance = cls.__new__(cls)
        for key, value in values.items():
            object.__setattr__(instance, key, value)
        return instance

    def model_dump(self) -> Dict[str, Any]:
        return {field: getattr(self, field) for field in getattr(self, "__annotations__", {})}

    def __repr__(self) -> str:  # pragma: no cover
        fields = ", ".join(f"{key}={value!r}" for key, value in self.model_dump().items())
        return f"{self.__class__.__name__}({fields})"

    def __eq__(self, other: object) -> bool:  # pragma: no cover - simple structural equality
        if not isinstance(other, BaseModel):
            return NotImplemented
        return self.model_dump() == other.model_dump()


def _collect_validators(cls: type) -> Dict[str, Dict[str, list[Callable[..., Any]]]]:
    validators: Dict[str, Dict[str, list[Callable[..., Any]]]] = {}
    for attribute in cls.__dict__.values():
        if callable(attribute) and hasattr(attribute, "__field_name__"):
            name = getattr(attribute, "__field_name__")
            mode = getattr(attribute, "__field_mode__", "after")
            validators.setdefault(name, {}).setdefault(mode, []).append(attribute)
    return validators


def _apply_validators(
    validators: Dict[str, Dict[str, list[Callable[..., Any]]]],
    field_name: str,
    value: Any,
    *,
    mode: str,
) -> Any:
    for validator in validators.get(field_name, {}).get(mode, []):
        value = validator(None, value)  # type: ignore[arg-type]
    return value


def _coerce(annotation: Any, value: Any) -> Any:
    origin = getattr(annotation, "__origin__", None)
    if origin is Optional and value is None:
        return None
    target = getattr(annotation, "__args__", [annotation])[0] if origin is Optional else annotation
    try:
        if target in (int, float, str, bool):
            return target(value)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Unable to coerce value '{value}' to {target}") from exc
    return value
