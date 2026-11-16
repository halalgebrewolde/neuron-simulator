from __future__ import annotations

import hashlib
import sys
from types import ModuleType
from typing import Any


def run_user_ai_source(source_code: str, st_api: Any) -> Any:
    """Execute the user's local AI app source and call its ``main(st)`` entrypoint."""
    digest = hashlib.sha256(source_code.encode("utf-8")).hexdigest()[:16]
    module_name = f"user_ai_{digest}"
    module = ModuleType(module_name)
    module.__file__ = "user_ai.py"
    module.__dict__["st"] = st_api
    sys.modules[module_name] = module

    try:
        compiled_source = compile(source_code, module.__file__, "exec")
        exec(compiled_source, module.__dict__)
        entrypoint = getattr(module, "main", None)
        if not callable(entrypoint):
            raise TypeError("Your AI source must define a callable main(st) function.")
        return entrypoint(st_api)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
