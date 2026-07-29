from __future__ import annotations


def full_tool_name(prefix: str, tool_name: str) -> str:
    if tool_name == prefix:
        return tool_name
    return f"{prefix}_{tool_name}"


def dispatch_plugin(plugins: list, full_name: str, args: dict) -> dict | None:
    for plugin in sorted(plugins, key=lambda item: len(item.PREFIX), reverse=True):
        prefix = plugin.PREFIX
        if full_name == prefix:
            return plugin.dispatch(prefix, args)

        marker = f"{prefix}_"
        if full_name.startswith(marker):
            return plugin.dispatch(full_name[len(marker):], args)

    return None
