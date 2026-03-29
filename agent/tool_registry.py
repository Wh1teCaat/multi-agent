from dataclasses import dataclass
from typing import Any

from SearchAgent import call_search_expert
from tools.task_tools import read_text_file, run_command, write_text_file


@dataclass(frozen=True)
class ToolPolicy:
    """工具策略定义（可集中治理）。"""

    allow_in_recall: bool = False
    require_confirm_param: bool = False
    confirm_param_name: str = "confirm"
    side_effect_level: str = "none"  # none | low | high


@dataclass(frozen=True)
class ToolSpec:
    """工具注册定义。"""

    name: str
    tool: Any
    policy: ToolPolicy
    description: str = ""


def build_default_tool_specs() -> list[ToolSpec]:
    """默认工具注册表。"""

    return [
        ToolSpec(
            name=call_search_expert.name,
            tool=call_search_expert,
            policy=ToolPolicy(
                allow_in_recall=False,
                require_confirm_param=False,
                side_effect_level="none",
            ),
            description="互联网搜索专家",
        ),
        ToolSpec(
            name=run_command.name,
            tool=run_command,
            policy=ToolPolicy(
                allow_in_recall=False,
                require_confirm_param=False,
                side_effect_level="high",
            ),
            description="本地命令执行",
        ),
        ToolSpec(
            name=read_text_file.name,
            tool=read_text_file,
            policy=ToolPolicy(
                allow_in_recall=False,
                require_confirm_param=False,
                side_effect_level="none",
            ),
            description="读取文件",
        ),
        ToolSpec(
            name=write_text_file.name,
            tool=write_text_file,
            policy=ToolPolicy(
                allow_in_recall=False,
                require_confirm_param=True,
                confirm_param_name="confirm",
                side_effect_level="high",
            ),
            description="写入文件",
        ),
    ]


def build_tool_maps(specs: list[ToolSpec]) -> tuple[list[Any], dict[str, Any], dict[str, ToolSpec]]:
    """根据注册表生成：
    - tools: 供 LLM bind_tools 使用
    - tools_by_name: 供执行层按名字调用
    - specs_by_name: 供策略层按名字做校验
    """

    tools = [spec.tool for spec in specs]
    tools_by_name = {spec.name: spec.tool for spec in specs}
    specs_by_name = {spec.name: spec for spec in specs}
    return tools, tools_by_name, specs_by_name
