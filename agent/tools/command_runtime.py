import shlex
import shutil
import sys


# 统一维护命令到 Python module 的回退映射。
# 当命令不在 PATH 中时，可回退为 `python -m <module>` 执行。
MODULE_FALLBACKS: dict[str, str] = {
    "pytest": "pytest",
    "ruff": "ruff",
    "pip": "pip",
}


SENSITIVE_PIP_ACTIONS = {
    "install",
    "uninstall",
    "remove",
    "upgrade",
}

# 通用高风险命令（可持续扩展）
SENSITIVE_COMMANDS = {
    "pip",
    "apt",
    "apt-get",
    "yum",
    "dnf",
    "apk",
    "brew",
    "docker",
    "kubectl",
    "systemctl",
    "service",
    "rm",
    "mv",
    "chmod",
    "chown",
    "kill",
    "pkill",
}

# 通用高风险动作词
SENSITIVE_ACTIONS = {
    "install",
    "uninstall",
    "remove",
    "upgrade",
    "delete",
    "reset",
    "prune",
    "purge",
    "reinstall",
    "downgrade",
    "restart",
    "stop",
    "start",
}


def _split_command_parts(command: str) -> list[str]:
    text = (command or "").strip()
    if not text:
        return []
    try:
        return shlex.split(text)
    except ValueError:
        return []


def _command_identity(parts: list[str]) -> tuple[str, list[str]]:
    """返回 (identity, args)。

    - 普通命令：identity=parts[0], args=parts[1:]
    - python -m xxx：identity=xxx, args=后续参数
    """
    if not parts:
        return "", []

    if len(parts) > 2 and parts[0] in {"python", "python3"} and parts[1] == "-m":
        return parts[2], parts[3:]

    return parts[0], parts[1:]


def resolve_executable_command(command: str, python_executable: str | None = None) -> str:
    """返回可执行命令。

    规则：
    1) 若主命令在 PATH 中，原样返回。
    2) 若主命令不在 PATH，但在 MODULE_FALLBACKS 中，回退为 `python -m <module>`。
    3) 其他情况保持原样，由调用方拿到真实错误码/错误信息。
    """
    text = (command or "").strip()
    if not text:
        return command

    try:
        parts = shlex.split(text)
    except ValueError:
        # 解析失败时不改写，交给执行阶段报错。
        return command

    if not parts:
        return command

    primary = parts[0]  # 主命令，如 pytest、python、ls 等

    # 已是 python -m 形式或 python 可执行，直接使用。
    if primary in {"python", "python3"}:
        return command

    if shutil.which(primary):
        return command

    module = MODULE_FALLBACKS.get(primary)
    if not module:  # 无映射，原样返回
        return command  

    py = python_executable or sys.executable    # 优先使用传入的解释器，否则用当前进程的解释器路径
    rest = shlex.join(parts[1:]) if len(parts) > 1 else ""
    if rest:
        return f"{py} -m {module} {rest}"
    return f"{py} -m {module}"


def requires_second_confirmation(command: str) -> bool:
    """是否需要二次确认。

    当前策略：
    1) 高风险命令（安装/卸载/系统服务/权限/删除类）触发二次确认。
    2) 对 pip 保留专项规则（安装、卸载、升级）。
    """
    parts = _split_command_parts(command)
    if not parts:
        return False

    identity, args = _command_identity(parts)
    if not identity:
        return False

    lowered_args = [a.lower() for a in args]
    action = lowered_args[0] if lowered_args else ""

    # pip 专项规则
    if identity == "pip" and action in SENSITIVE_PIP_ACTIONS:
        return True

    # 通用高风险命令
    if identity in SENSITIVE_COMMANDS:
        return True

    # 通用高风险动作词
    if action in SENSITIVE_ACTIONS:
        return True

    return False


def command_exit_hint(command: str, returncode: int) -> str:
    """给返回码生成可读提示。"""
    normalized = command.strip()

    # 针对 pytest 保留更精细解释
    if normalized.startswith("pytest") or normalized.startswith("python -m pytest"):
        pytest_mapping = {
            0: "命令执行成功：pytest 全部通过。",
            1: "命令执行完成，但 pytest 存在失败用例。",
            2: "命令被中断（pytest interrupted）。",
            3: "命令执行失败：pytest 内部错误。",
            4: "命令执行失败：pytest 参数使用错误。",
            5: "命令执行完成，但 pytest 未收集到测试（no tests ran）。",
        }
        return pytest_mapping.get(returncode, f"命令返回码 {returncode}。")

    common_mapping = {
        0: "命令执行成功。",
        1: "命令执行失败（通用错误）。",
        2: "命令使用错误（例如参数不正确）。",
        126: "命令找到但不可执行（权限或格式问题）。",
        127: "命令未找到（command not found）。",
        130: "命令被中断（通常是 Ctrl+C / SIGINT）。",
        137: "命令被强制终止（通常是 SIGKILL，可能因资源限制）。",
        143: "命令被终止（SIGTERM）。",
    }

    if returncode in common_mapping:
        return common_mapping[returncode]

    if returncode > 128:
        signal_no = returncode - 128
        return f"命令异常终止，可能由信号 {signal_no} 导致（返回码 {returncode}）。"

    return f"命令返回码 {returncode}。"
