from __future__ import annotations

import fnmatch
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


LOCAL_TOOL_NAMES = ("Read", "Write", "Edit", "Bash", "Grep", "Glob", "Skill")

LOCAL_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "Read",
        "description": "Read a UTF-8 file from the task workspace or the read-only L1 skill catalog.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "Write",
        "description": "Write a UTF-8 file inside the task workspace.",
        "input_schema": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "Edit",
        "description": "Replace exact text in a UTF-8 file inside the task workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
    {
        "name": "Grep",
        "description": "Search text files in the task workspace or read-only L1 skill catalog.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "glob": {"type": "string"},
                "case_insensitive": {"type": "boolean"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "Glob",
        "description": "List files matching a glob in the task workspace or read-only L1 skill catalog.",
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
            "required": ["pattern"],
        },
    },
    {
        "name": "Skill",
        "description": "Read one named tool-level (L1) skill.",
        "input_schema": {
            "type": "object",
            "properties": {"skill": {"type": "string"}, "name": {"type": "string"}},
        },
    },
    {
        "name": "Bash",
        "description": "Run a restricted file-oriented command in the task workspace.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
]

_BASH_COMMANDS = {
    "pwd",
    "ls",
    "find",
    "cat",
    "head",
    "tail",
    "grep",
    "wc",
    "stat",
    "mkdir",
    "cp",
    "base64",
    "realpath",
    "readlink",
    "test",
    "echo",
    "cd",
}
_BASH_FORBIDDEN = {
    "curl",
    "wget",
    "python",
    "python3",
    "perl",
    "ruby",
    "node",
    "sudo",
    "rm",
    "chmod",
    "chown",
    "kill",
    "pkill",
    "nohup",
    "apt",
    "apt-get",
    "pip",
    "pip3",
    "conda",
}
_BASH_UNSAFE_TEXT = re.compile(r"(?:`|\$\(|\$\{|;|&&|\|\||\n|\r)")
_ARTIFACT_REF = re.compile(r"^<artifact:(.+)>$")


class LocalToolError(ValueError):
    pass


def is_local_tool(tool_name: str) -> bool:
    return tool_name in LOCAL_TOOL_NAMES


class LocalToolExecutor:
    """Per-task, filesystem-confined implementations of supported local tools."""

    def __init__(self, workspace: str | Path, l1_skills_root: str | Path) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.l1_skills_root = Path(l1_skills_root).expanduser().resolve()

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        try:
            if not isinstance(arguments, dict):
                raise LocalToolError("`arguments` must be an object")
            handler = getattr(self, f"_run_{tool_name.lower()}", None)
            if handler is None or tool_name not in LOCAL_TOOL_NAMES:
                raise LocalToolError(f"unsupported local tool: {tool_name}")
            result = handler(arguments)
            return self._result(tool_name, result=result, started=started)
        except Exception as exc:
            return self._result(tool_name, error=exc, started=started)

    def _result(
        self,
        tool_name: str,
        *,
        result: Any = None,
        error: Exception | None = None,
        started: float,
    ) -> dict[str, Any]:
        ok = error is None
        return {
            "ok": ok,
            "tool_name": tool_name,
            "result": result if ok else None,
            "error": None
            if ok
            else {"type": type(error).__name__, "message": str(error)},
            "latency_sec": round(time.monotonic() - started, 6),
            "transport_ok": True,
            "tool_schema_valid": True,
            "tool_execution_success": ok,
            "tool_semantic_success": ok,
            "semantic_unknown": False,
            "metadata": {"executor": "local_sandbox", "workspace": self.workspace.name},
        }

    @staticmethod
    def _require_string(arguments: dict[str, Any], key: str) -> str:
        value = arguments.get(key)
        if not isinstance(value, str) or not value:
            raise LocalToolError(f"`{key}` must be a non-empty string")
        return value

    def _path(self, raw: str, *, write: bool = False) -> Path:
        artifact_match = _ARTIFACT_REF.fullmatch(raw.strip())
        if artifact_match:
            raw = artifact_match.group(1)
            if raw.startswith("local/"):
                raw = raw[len("local/") :]

        normalized = raw.replace("\\", "/")
        skill_prefix = "skills/L1_tools/"
        if normalized == "skills/L1_tools":
            candidate = self.l1_skills_root
            is_skill = True
        elif normalized.startswith(skill_prefix):
            candidate = self.l1_skills_root / normalized[len(skill_prefix) :]
            is_skill = True
        else:
            p = Path(raw).expanduser()
            candidate = p if p.is_absolute() else self.workspace / p
            is_skill = False

        resolved = candidate.resolve(strict=False)
        root = self.l1_skills_root if is_skill else self.workspace
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise LocalToolError("path escapes the permitted workspace") from exc
        if write and is_skill:
            raise LocalToolError("L1 skills are read-only")

        # Existing symlinks must also resolve inside their permitted root.
        probe = candidate
        while probe != root and not probe.exists():
            probe = probe.parent
        if probe.exists():
            try:
                probe.resolve().relative_to(root)
            except ValueError as exc:
                raise LocalToolError("symlink escapes the permitted workspace") from exc
        return resolved

    def _display(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.workspace))
        except ValueError:
            return f"skills/L1_tools/{path.relative_to(self.l1_skills_root)}"

    def _run_read(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._path(self._require_string(arguments, "file_path"))
        if not path.is_file():
            raise LocalToolError("file does not exist")
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        offset = arguments.get("offset", 0)
        limit = arguments.get("limit")
        if not isinstance(offset, int) or offset < 0:
            raise LocalToolError("`offset` must be a non-negative integer")
        if limit is not None and (not isinstance(limit, int) or limit < 0):
            raise LocalToolError("`limit` must be a non-negative integer")
        selected = lines[offset:] if limit is None else lines[offset : offset + limit]
        return {"path": self._display(path), "content": "".join(selected)}

    def _run_write(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._path(self._require_string(arguments, "file_path"), write=True)
        content = arguments.get("content")
        if not isinstance(content, str):
            raise LocalToolError("`content` must be a string")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {"path": self._display(path), "bytes_written": len(content.encode("utf-8"))}

    def _run_edit(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = self._path(self._require_string(arguments, "file_path"), write=True)
        old = self._require_string(arguments, "old_string")
        new = arguments.get("new_string")
        if not isinstance(new, str):
            raise LocalToolError("`new_string` must be a string")
        if not path.is_file():
            raise LocalToolError("file does not exist")
        text = path.read_text(encoding="utf-8")
        count = text.count(old)
        if count == 0:
            raise LocalToolError("`old_string` was not found")
        replace_all = arguments.get("replace_all") is True
        if count > 1 and not replace_all:
            raise LocalToolError("`old_string` is not unique; use replace_all=true")
        updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        path.write_text(updated, encoding="utf-8")
        return {"path": self._display(path), "replacements": count if replace_all else 1}

    def _search_root(self, arguments: dict[str, Any]) -> Path:
        raw = arguments.get("path", ".")
        if not isinstance(raw, str):
            raise LocalToolError("`path` must be a string")
        return self._path(raw)

    def _run_glob(self, arguments: dict[str, Any]) -> dict[str, Any]:
        root = self._search_root(arguments)
        pattern = self._require_string(arguments, "pattern")
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise LocalToolError("glob pattern may not escape its search root")
        if not root.exists():
            return {"matches": []}
        matches = []
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            safe_path = self._path(self._display(path))
            matches.append(self._display(safe_path))
        matches.sort()
        return {"matches": matches}

    def _run_grep(self, arguments: dict[str, Any]) -> dict[str, Any]:
        root = self._search_root(arguments)
        pattern = self._require_string(arguments, "pattern")
        flags = re.IGNORECASE if arguments.get("case_insensitive") is True else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            raise LocalToolError(f"invalid regex: {exc}") from exc
        file_glob = arguments.get("glob", "*")
        if not isinstance(file_glob, str):
            raise LocalToolError("`glob` must be a string")
        if Path(file_glob).is_absolute() or ".." in Path(file_glob).parts:
            raise LocalToolError("file glob may not escape its search root")
        files = [root] if root.is_file() else root.rglob("*")
        matches: list[dict[str, Any]] = []
        for path in files:
            if not path.is_file() or not fnmatch.fnmatch(path.name, file_glob):
                continue
            try:
                safe_path = self._path(self._display(path))
                lines = safe_path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError, LocalToolError):
                continue
            for line_no, line in enumerate(lines, start=1):
                if regex.search(line):
                    matches.append({"path": self._display(safe_path), "line": line_no, "text": line})
        return {"matches": matches}

    def _run_skill(self, arguments: dict[str, Any]) -> dict[str, Any]:
        name = arguments.get("skill") or arguments.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
            raise LocalToolError("`skill` must name one L1 skill")
        path = self._path(f"skills/L1_tools/{name}/SKILL.md")
        if not path.is_file():
            raise LocalToolError(f"L1 skill not found: {name}")
        return {
            "skill": name,
            "path": f"skills/L1_tools/{name}/SKILL.md",
            "content": path.read_text(encoding="utf-8"),
        }

    def _run_bash(self, arguments: dict[str, Any]) -> dict[str, Any]:
        command = self._require_string(arguments, "command").strip()
        if _BASH_UNSAFE_TEXT.search(command):
            raise LocalToolError("shell expansion, chaining, and command substitution are forbidden")
        lexer = shlex.shlex(command, posix=True, punctuation_chars="|><&;")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
        if not tokens:
            raise LocalToolError("empty command")

        commands, redirect = self._split_pipeline(tokens)
        stdin: str | None = None
        stdout = ""
        for command_tokens in commands:
            stdout = self._run_bash_command(command_tokens, stdin)
            stdin = stdout
        if redirect is not None:
            mode, raw_path = redirect
            path = self._path(raw_path, write=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open(mode, encoding="utf-8") as handle:
                handle.write(stdout)
            return {"stdout": "", "artifact": self._display(path)}
        return {"stdout": stdout}

    def _split_pipeline(
        self, tokens: list[str]
    ) -> tuple[list[list[str]], tuple[str, str] | None]:
        commands: list[list[str]] = [[]]
        redirect: tuple[str, str] | None = None
        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            if token == "|":
                if not commands[-1]:
                    raise LocalToolError("invalid pipeline")
                commands.append([])
            elif token in {">", ">>"}:
                if idx != len(tokens) - 2 or redirect is not None:
                    raise LocalToolError("redirection is only allowed once at the end")
                redirect = ("a" if token == ">>" else "w", tokens[idx + 1])
                break
            elif token in {"<", "2>", "2>>", "&"}:
                raise LocalToolError("unsupported shell redirection")
            else:
                commands[-1].append(token)
            idx += 1
        if any(not item for item in commands):
            raise LocalToolError("invalid pipeline")
        return commands, redirect

    def _run_bash_command(self, tokens: list[str], stdin: str | None) -> str:
        name = tokens[0]
        if name in _BASH_FORBIDDEN or name not in _BASH_COMMANDS:
            raise LocalToolError(f"command is not permitted: {name}")
        if name == "find" and any(item in {"-exec", "-execdir", "-delete", "-ok", "-okdir"} for item in tokens):
            raise LocalToolError("dangerous find action is forbidden")
        self._validate_bash_options(name, tokens[1:])
        if name == "find":
            self._validate_find_arguments(tokens)
        if name == "cd":
            if len(tokens) != 2:
                raise LocalToolError("cd requires exactly one path")
            path = self._path(tokens[1])
            if not path.is_dir():
                raise LocalToolError("cd target is not a directory")
            return self._display(path) + "\n"
        if name == "test":
            if len(tokens) == 2:
                pass
            elif len(tokens) == 3 and tokens[1] in {"-e", "-f", "-d", "-r", "-w", "-s", "-L"}:
                pass
            else:
                raise LocalToolError("test supports only one confined file operand")

        rewritten = [name]
        for idx, token in enumerate(tokens[1:], start=1):
            if self._bash_token_is_path(name, tokens, idx):
                resolved = self._path(token, write=name in {"mkdir"} or (name == "cp" and idx == len(tokens) - 1))
                rewritten.append(str(resolved))
            else:
                rewritten.append(token)

        executable = next(
            (candidate for candidate in (Path("/usr/bin") / name, Path("/bin") / name) if candidate.is_file()),
            None,
        )
        if executable is None:
            raise LocalToolError(f"command is unavailable: {name}")
        completed = subprocess.run(
            [str(executable), *rewritten[1:]],
            cwd=self.workspace,
            input=stdin,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or f"{name} exited {completed.returncode}"
            raise LocalToolError(self._sanitize_output(message))
        return self._sanitize_output(completed.stdout)

    @staticmethod
    def _validate_find_arguments(tokens: list[str]) -> None:
        if len(tokens) < 2:
            raise LocalToolError("find requires one confined search root")
        # Exactly one search root is supported. Every later token must be one
        # of the small expression forms advertised by the sandbox.
        cursor = 2
        value_options = {"-name", "-type", "-maxdepth", "-mindepth"}
        while cursor < len(tokens):
            option = tokens[cursor]
            if option == "-print":
                cursor += 1
                continue
            if option not in value_options or cursor + 1 >= len(tokens):
                raise LocalToolError("find supports one root plus -name/-type/-maxdepth/-mindepth/-print")
            cursor += 2

    def _sanitize_output(self, text: str) -> str:
        return text.replace(str(self.l1_skills_root), "skills/L1_tools").replace(
            str(self.workspace), "."
        )

    @staticmethod
    def _validate_bash_options(name: str, args: list[str]) -> None:
        allowed_options = {
            "pwd": set(),
            "ls": {"-a", "-l", "-h", "-la", "-al", "-lh", "-hl", "-lah", "-alh"},
            "cat": {"-n"},
            "head": {"-n", "-c"},
            "tail": {"-n", "-c"},
            "grep": {"-i", "-n", "-v", "-E", "-F", "-e"},
            "wc": {"-l", "-w", "-m", "-c"},
            "stat": {"-c"},
            "mkdir": {"-p"},
            "cp": {"-r", "-R", "-f"},
            "base64": {"-d", "--decode", "-w"},
            "realpath": {"-e", "-m"},
            "readlink": {"-f", "-e", "-m"},
            "test": {"-e", "-f", "-d", "-r", "-w", "-s", "-L"},
            "echo": {"-n"},
            "cd": set(),
            "find": {"-name", "-type", "-maxdepth", "-mindepth", "-print"},
        }
        for token in args:
            if token.startswith("--") and token not in allowed_options[name]:
                raise LocalToolError(f"option is not permitted for {name}: {token}")
            if token.startswith("-") and token != "-" and token not in allowed_options[name]:
                # Numeric values such as `head -n 5` are not options.
                try:
                    int(token)
                    continue
                except ValueError:
                    raise LocalToolError(f"option is not permitted for {name}: {token}")

    @staticmethod
    def _bash_token_is_path(name: str, tokens: list[str], idx: int) -> bool:
        token = tokens[idx]
        if token.startswith("-"):
            return False
        if name == "echo":
            return False
        if name == "test":
            return not token.startswith("-")
        if name == "grep":
            cursor = 1
            explicit_pattern = False
            while cursor < len(tokens):
                if tokens[cursor] == "-e":
                    explicit_pattern = True
                    cursor += 2
                    continue
                if tokens[cursor].startswith("-"):
                    cursor += 1
                    continue
                return idx >= cursor if explicit_pattern else idx > cursor
            return False
        if name in {"head", "tail"} and idx > 1 and tokens[idx - 1] in {"-n", "-c"}:
            return False
        if name == "find":
            return idx == 1
        if name == "base64":
            return not (tokens[idx - 1] == "-w")
        if name == "stat" and tokens[idx - 1] == "-c":
            return False
        if name == "test":
            return not token.startswith("-")
        return True
