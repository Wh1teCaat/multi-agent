import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tools.command_runtime import command_exit_hint, resolve_executable_command, requires_second_confirmation


ALLOWED_PREFIXES = [
    "pytest",
    "go test",
    "ruff",
    "python -m",
    "python3 -m",
    "pip",
    "git add",
    "git commit",
    "git push",
    "git status",
    "ls",
    "cat",
    "grep",
    "head",
    "tail",
]



class CommandRunResult(BaseModel):
    """`run_command` 的统一结构化输出。"""

    ok: bool = Field(description="命令是否成功执行（通常等价于 returncode == 0）。")
    command: str = Field(description="用户请求执行的原始命令。")
    executed_command: str = Field(description="实际执行的命令（可能经过回退改写）。")

    dry_run: bool = Field(description="是否 dry-run 模式。")
    executed: bool = Field(description="是否真正执行了命令。")

    visible_in_terminal: bool = Field(default=False, description="是否在当前终端实时输出。")
    open_new_terminal: bool = Field(default=False, description="是否尝试在新终端执行。")

    returncode: int | None = Field(default=None, description="进程返回码。")
    hint: str | None = Field(default=None, description="对返回码的可读解释。")

    stdout: str | None = Field(default=None, description="标准输出。")
    stderr: str | None = Field(default=None, description="标准错误输出。")

    pid: int | None = Field(default=None, description="子进程 PID（如有）。")
    message: str | None = Field(default=None, description="补充说明。")
    error: str | None = Field(default=None, description="错误信息（若有）。")


def _result_dict(**kwargs: Any) -> dict[str, Any]:
    """按 schema 校验，并返回字典形式结果。"""
    return CommandRunResult(**kwargs).model_dump()


async def interactive_yes_no(prompt: str) -> bool | None:
    """通用 y/n 交互确认。

    返回：
    - True: 用户确认
    - False: 用户拒绝
    - None: 当前环境不支持交互输入
    """
    if not sys.stdin or not sys.stdin.isatty():
        return None

    try:
        answer = await asyncio.to_thread(input, prompt)
    except Exception:
        return None

    normalized = (answer or "").strip().lower()
    return normalized in {"y", "yes", "是", "ok", "confirm"}


async def _interactive_confirm_if_needed(command: str) -> bool | None:
    """在当前终端进行 y/n 二次确认。

    返回：
    - True: 用户确认执行
    - False: 用户拒绝执行
    - None: 当前环境不支持交互输入
    """
    if not sys.stdin or not sys.stdin.isatty():
        return None

    prompt = (
        f"[agent] 检测到高风险命令：{command}\n"
        "[agent] 是否继续执行？(y/n): "
    )
    return await interactive_yes_no(prompt)


async def _stream_and_capture(stream: asyncio.StreamReader | None, *, is_stderr: bool = False) -> str:
    """实时转发子进程输出到当前终端，并返回完整文本。"""
    if stream is None:
        return ""

    chunks: list[str] = []
    while True:
        line = await stream.readline()
        if not line:
            break

        text = line.decode("utf-8", errors="ignore")
        chunks.append(text)

        if is_stderr:
            sys.stderr.write(text)
            sys.stderr.flush()
        else:
            sys.stdout.write(text)
            sys.stdout.flush()

    return "".join(chunks)


def _is_allowed(command: str) -> bool:
    cmd = command.strip()
    return any(cmd.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def _resolve_text_path(path: str) -> Path:
    """优先按当前工作目录解析；不存在时回退到 agent 根目录。"""
    raw = Path(path)
    if raw.is_absolute():
        return raw

    cwd_path = (Path.cwd() / raw).resolve()
    if cwd_path.exists():
        return cwd_path

    agent_root = Path(__file__).resolve().parents[1]
    agent_path = (agent_root / raw).resolve()
    return agent_path


@tool
async def run_command(
    command: str,
    dry_run: bool = True,
    confirm: bool = False,
    visible_in_terminal: bool = False,
    open_new_terminal: bool = False,
) -> dict[str, Any]:
    """执行本地命令。

    - dry_run=True: 仅返回计划，不执行。
    - confirm=True: 高风险命令（二次确认）允许真正执行。
    - visible_in_terminal=True: 在当前运行进程的终端直接展示执行过程。
    - open_new_terminal=True: 尝试打开系统新终端执行。
    """
    exec_command = resolve_executable_command(command)

    if not _is_allowed(command):
        return _result_dict(
            ok=False,
            command=command,
            executed_command=exec_command,
            dry_run=dry_run,
            executed=False,
            visible_in_terminal=visible_in_terminal,
            open_new_terminal=open_new_terminal,
            error=f"command not allowed: {command}",
        )

    if dry_run:
        return _result_dict(
            ok=True,
            command=command,
            executed_command=exec_command,
            dry_run=True,
            executed=False,
            visible_in_terminal=visible_in_terminal,
            open_new_terminal=open_new_terminal,
            message="dry-run 模式：命令未实际执行",
        )

    if requires_second_confirmation(exec_command) and not confirm:
        interactive = await _interactive_confirm_if_needed(exec_command)
        if interactive is True:
            # 用户在终端确认后，自动释放 confirm
            confirm = True
        elif interactive is False:
            return _result_dict(
                ok=False,
                command=command,
                executed_command=exec_command,
                dry_run=False,
                executed=False,
                visible_in_terminal=visible_in_terminal,
                open_new_terminal=open_new_terminal,
                message="用户拒绝执行高风险命令，任务已终止。",
                error="user rejected sensitive command",
            )
        else:
            return _result_dict(
                ok=False,
                command=command,
                executed_command=exec_command,
                dry_run=False,
                executed=False,
                visible_in_terminal=visible_in_terminal,
                open_new_terminal=open_new_terminal,
                message="检测到高风险命令。当前环境不支持交互确认，请设置 confirm=true 后重试。",
                error="second confirmation required",
            )

    if open_new_terminal:
        terminal_bin = os.getenv("AGENT_TERMINAL_BIN", "x-terminal-emulator")
        shell_cmd = f"echo '[agent] running: {exec_command}'; {exec_command}; echo; echo '[agent] done'; exec bash"
        try:
            proc = await asyncio.create_subprocess_exec(
                terminal_bin,
                "-e",
                "bash",
                "-lc",
                shell_cmd,
            )
            return _result_dict(
                ok=True,
                command=command,
                executed_command=exec_command,
                dry_run=False,
                executed=True,
                visible_in_terminal=True,
                open_new_terminal=True,
                pid=proc.pid,
                message=f"已尝试在新终端执行（{terminal_bin}）。",
            )
        except Exception as e:
            return _result_dict(
                ok=False,
                command=command,
                executed_command=exec_command,
                dry_run=False,
                executed=False,
                visible_in_terminal=True,
                open_new_terminal=True,
                error=f"打开新终端失败: {e}",
            )

    if visible_in_terminal:
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-lc",
            f"set -x; {exec_command}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout_task = asyncio.create_task(_stream_and_capture(proc.stdout, is_stderr=False))
        stderr_task = asyncio.create_task(_stream_and_capture(proc.stderr, is_stderr=True))

        returncode, stdout_text, stderr_text = await asyncio.gather(
            proc.wait(),
            stdout_task,
            stderr_task,
        )

        return _result_dict(
            ok=returncode == 0,
            command=command,
            executed_command=exec_command,
            dry_run=False,
            executed=True,
            visible_in_terminal=True,
            open_new_terminal=False,
            returncode=returncode,
            hint=command_exit_hint(command, returncode),
            stdout=stdout_text,
            stderr=stderr_text,
        )

    proc = await asyncio.create_subprocess_shell(
        exec_command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return _result_dict(
        ok=proc.returncode == 0,
        command=command,
        executed_command=exec_command,
        dry_run=False,
        executed=True,
        visible_in_terminal=False,
        open_new_terminal=False,
        returncode=proc.returncode,
        hint=command_exit_hint(command, proc.returncode),
        stdout=stdout.decode("utf-8", errors="ignore"),
        stderr=stderr.decode("utf-8", errors="ignore"),
    )


@tool
async def read_text_file(path: str) -> dict[str, Any]:
    """读取文本文件。"""
    try:
        resolved = _resolve_text_path(path)
        with open(resolved, "r", encoding="utf-8") as f:
            return {"ok": True, "path": str(resolved), "content": f.read()}
    except Exception as e:
        return {"ok": False, "path": path, "error": str(e), "cwd": str(Path.cwd())}


@tool
async def write_text_file(path: str, content: str, confirm: bool = False, append: bool = False) -> dict[str, Any]:
    """写入文本文件。需显式 confirm=true；若父目录不存在会自动创建；支持 append 追加模式。"""
    if not confirm:
        return {"ok": False, "path": path, "error": "write denied: confirm=true required"}
    try:
        resolved = _resolve_text_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(resolved, mode, encoding="utf-8") as f:
            f.write(content)
        return {"ok": True, "path": str(resolved), "append": append}
    except Exception as e:
        return {"ok": False, "path": path, "error": str(e)}
