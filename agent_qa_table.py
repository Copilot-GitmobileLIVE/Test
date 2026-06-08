#This scripts get Agent name and a question and then invoke the agent and get the respocse back
#output a table including the agentname, Question,Answer, and tools called
# python .\agent_qa_table.py --agent-name "Web Q&A Agent" --question "Where is London" --wait-seconds 30
import argparse
import csv
import json
import io
import os
import subprocess
import sys
import time
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Event:
    event_type: str
    timestamp: str
    data: dict[str, Any]


@dataclass
class AgentCallRow:
    agent_name: str
    request: str
    response: str
    tools: list[str]


def normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def shorten(text: str, limit: int = 240) -> str:
    flat = " ".join((text or "").split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 3] + "..."


def default_workspace_storage() -> Path:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        raise RuntimeError("APPDATA is not set")
    return Path(appdata) / "Code" / "User" / "workspaceStorage"


def find_transcript_files(workspace_storage: Path) -> list[Path]:
    files: list[Path] = []
    for root in workspace_storage.rglob("GitHub.copilot-chat"):
        tdir = root / "transcripts"
        if tdir.is_dir():
            files.extend(sorted(tdir.glob("*.jsonl")))
    return sorted(files)


def find_chat_session_files(workspace_storage: Path) -> list[Path]:
    files: list[Path] = []
    for root in workspace_storage.rglob("chatSessions"):
        if root.is_dir():
            files.extend(sorted(root.glob("*.jsonl")))
    return sorted(files)


def parse_events(files: list[Path]) -> list[Event]:
    events: list[Event] = []
    for file_path in files:
        try:
            with file_path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    data = obj.get("data")
                    if not isinstance(data, dict):
                        data = {}
                    events.append(
                        Event(
                            event_type=str(obj.get("type", "")),
                            timestamp=str(obj.get("timestamp", "")),
                            data=data,
                        )
                    )
        except OSError:
            continue

    events.sort(key=lambda e: e.timestamp)
    return events


def extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = extract_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()

    if isinstance(value, dict):
        for key in ("content", "message", "text", "value"):
            text = extract_text(value.get(key))
            if text:
                return text
    return ""


def extract_tool_names_from_request(data: dict[str, Any]) -> list[str]:
    tool_requests = data.get("toolRequests")
    if not isinstance(tool_requests, list):
        return []

    tools: list[str] = []
    for request in tool_requests:
        if not isinstance(request, dict):
            continue
        name = str(request.get("name", "")).strip()
        if name:
            tools.append(name)
    return tools


def merge_tools(tools: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for tool in tools:
        if tool and tool not in seen:
            seen.add(tool)
            deduped.append(tool)
    return deduped


def resolve_code_cli() -> str | None:
    # Try PATH first.
    for candidate in ("code", "code.cmd", "Code.exe"):
        hit = shutil.which(candidate)
        if hit:
            return hit

    # Try common Windows install locations.
    local_appdata = os.environ.get("LOCALAPPDATA", "")
    program_files = os.environ.get("ProgramFiles", "")
    program_files_x86 = os.environ.get("ProgramFiles(x86)", "")

    candidates = [
        Path(local_appdata) / "Programs" / "Microsoft VS Code" / "bin" / "code.cmd",
        Path(local_appdata) / "Programs" / "Microsoft VS Code" / "Code.exe",
        Path(program_files) / "Microsoft VS Code" / "bin" / "code.cmd",
        Path(program_files) / "Microsoft VS Code" / "Code.exe",
        Path(program_files_x86) / "Microsoft VS Code" / "bin" / "code.cmd",
        Path(program_files_x86) / "Microsoft VS Code" / "Code.exe",
    ]

    for p in candidates:
        if p and p.exists():
            return str(p)

    return None


def extract_tool_name(value: dict[str, Any]) -> str:
    tool_id = str(value.get("toolId", "")).strip()
    if tool_id:
        return tool_id

    tool_name = str(value.get("toolName", "")).strip()
    if tool_name:
        return tool_name

    return extract_text(value.get("invocationMessage"))


def parse_chat_session_file(file_path: Path) -> dict[int, dict[str, Any]]:
    requests: dict[int, dict[str, Any]] = {}

    try:
        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not isinstance(obj, dict):
                    continue

                path = obj.get("k")
                value = obj.get("v")
                if not isinstance(path, list) or len(path) < 3 or path[0] != "requests":
                    continue

                try:
                    request_index = int(path[1])
                except (TypeError, ValueError):
                    continue

                record = requests.setdefault(
                    request_index,
                    {"question": "", "answer": "", "tools": [], "agent_calls": []},
                )

                if len(path) >= 3 and path[2] == "result" and isinstance(value, dict):
                    metadata = value.get("metadata")
                    if isinstance(metadata, dict):
                        rendered = metadata.get("renderedUserMessage")
                        if isinstance(rendered, list):
                            parts: list[str] = []
                            for item in rendered:
                                if isinstance(item, dict):
                                    text = str(item.get("text", "")).strip()
                                    if text:
                                        parts.append(text)
                            if parts:
                                record["question"] = "\n".join(parts)

                if len(path) >= 3 and path[2] == "response" and isinstance(value, list):
                    response_texts: list[str] = []
                    response_tools: list[str] = []
                    subagent_calls: dict[str, AgentCallRow] = {}
                    for item in value:
                        if not isinstance(item, dict):
                            continue

                        if item.get("kind") == "toolInvocationSerialized":
                            tool_name = extract_tool_name(item)
                            tool_call_id = str(item.get("toolCallId", "")).strip()
                            subagent_invocation_id = str(item.get("subAgentInvocationId", "")).strip()
                            tool_specific_data = item.get("toolSpecificData")

                            if isinstance(tool_specific_data, dict) and tool_specific_data.get("kind") == "subagent":
                                subagent_calls[tool_call_id] = AgentCallRow(
                                    agent_name=str(tool_specific_data.get("agentName", "")).strip() or "(unknown agent)",
                                    request=str(tool_specific_data.get("prompt", "")).strip()
                                    or extract_text(item.get("invocationMessage")),
                                    response=str(tool_specific_data.get("result", "")).strip()
                                    or "(answer not found yet)",
                                    tools=[],
                                )
                                continue

                            if subagent_invocation_id and subagent_invocation_id in subagent_calls:
                                if tool_name:
                                    subagent_calls[subagent_invocation_id].tools.append(tool_name)
                                continue

                            if tool_name:
                                response_tools.append(tool_name)
                        else:
                            text = extract_text(item)
                            if text:
                                response_texts.append(text)

                    if response_texts and not record.get("answer"):
                        record["answer"] = "\n".join(response_texts)
                    if response_tools:
                        record_tools = record.setdefault("tools", [])
                        if isinstance(record_tools, list):
                            record_tools.extend(response_tools)

                    if subagent_calls:
                        record["agent_calls"] = list(subagent_calls.values())
    except OSError:
        return {}

    return requests


def find_chat_session_interaction(
    workspace_storage: Path,
    question: str,
    question_variants: list[str] | None = None,
) -> tuple[str, list[AgentCallRow]]:
    q_norm = normalize(question)
    variant_norms = {q_norm}
    if question_variants:
        variant_norms.update(normalize(v) for v in question_variants if v)

    for file_path in find_chat_session_files(workspace_storage):
        requests = parse_chat_session_file(file_path)
        for record in requests.values():
            session_question = str(record.get("question", ""))
            session_norm = normalize(session_question)
            if not session_norm:
                continue
            if session_norm in variant_norms or any(v and v in session_norm for v in variant_norms) or (q_norm and q_norm in session_norm):
                agent_calls = record.get("agent_calls", [])
                if isinstance(agent_calls, list):
                    rows = [row for row in agent_calls if isinstance(row, AgentCallRow)]
                    if rows:
                        for row in rows:
                            row.tools = merge_tools(row.tools)
                        return question, rows

                answer = str(record.get("answer", "")).strip() or "(answer not found yet)"
                tools = record.get("tools", [])
                if not isinstance(tools, list):
                    tools = []
                return question, [
                    AgentCallRow(
                        agent_name="",
                        request=question,
                        response=answer,
                        tools=merge_tools([str(tool) for tool in tools]),
                    )
                ]

    return question, []


def invoke_agent_via_cli(agent_name: str, question: str) -> tuple[str, str]:
    # Best-effort invocation through VS Code CLI chat command.
    prompt = f"@{agent_name} {question}" if agent_name else question
    cli = resolve_code_cli()
    if not cli:
        return prompt, "VS Code CLI not found. Install 'code' command or ensure VS Code is installed in a standard path."

    cmd = [cli, "chat", "--mode", "agent", "--reuse-window", prompt]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        stderr_text = (proc.stderr or "").strip()
        if proc.returncode != 0 and stderr_text:
            return prompt, f"CLI returned {proc.returncode}: {stderr_text}"
        return prompt, ""
    except FileNotFoundError:
        return prompt, "'code' CLI not found in PATH"
    except Exception as exc:  # pragma: no cover - defensive
        return prompt, str(exc)


def find_interaction(
    events: list[Event],
    question: str,
    question_variants: list[str] | None = None,
    since_timestamp: str | None = None,
) -> AgentCallRow:
    q_norm = normalize(question)
    variant_norms = {q_norm}
    if question_variants:
        variant_norms.update(normalize(v) for v in question_variants if v)

    user_idx = -1
    contains_idx = -1
    for i in range(len(events) - 1, -1, -1):
        ev = events[i]
        if ev.event_type != "user.message":
            continue
        if since_timestamp and ev.timestamp < since_timestamp:
            continue
        content = str(ev.data.get("content", ""))
        content_norm = normalize(content)
        if content_norm in variant_norms:
            user_idx = i
            break
        if q_norm and q_norm in content_norm and contains_idx == -1:
            contains_idx = i

    if user_idx == -1 and contains_idx != -1:
        user_idx = contains_idx

    if user_idx == -1:
        return AgentCallRow(
            agent_name="",
            request=question,
            response="(not found yet: invocation did not produce a matching user.message in transcripts)",
            tools=[],
        )

    answer = "(answer not found yet)"
    tools: list[str] = []

    for i in range(user_idx + 1, len(events)):
        ev = events[i]
        if ev.event_type == "user.message":
            break

        if ev.event_type == "assistant.message" and answer == "(answer not found yet)":
            text = extract_text(ev.data)
            if text:
                answer = text

            tools.extend(extract_tool_names_from_request(ev.data))

        if ev.event_type == "tool.execution_start":
            tool_name = str(ev.data.get("toolName", "")).strip()
            if tool_name:
                tools.append(tool_name)

    # Keep order, remove duplicates.
    return AgentCallRow(
        agent_name="",
        request=question,
        response=answer,
        tools=merge_tools(tools),
    )


def markdown_table(rows: list[AgentCallRow]) -> str:
    def esc(value: str) -> str:
        single_line = " ".join(value.split())
        return single_line.replace("|", "\\|")

    lines = [
        "| Agent Called | Request | Response | Tools Called |",
        "| --- | --- | --- | --- |",
    ]

    for row in rows:
        tools_cell = ", ".join(row.tools) if row.tools else "(none detected)"
        lines.append(
            f"| {esc(row.agent_name or '(unknown/direct)')} | {esc(row.request)} | {esc(row.response)} | {esc(tools_cell)} |"
        )

    return "\n".join(lines)


def rows_as_dicts(rows: list[AgentCallRow]) -> list[dict[str, Any]]:
    return [
        {
            "agent_called": row.agent_name or "(unknown/direct)",
            "request": row.request,
            "response": row.response,
            "tools_called": row.tools,
        }
        for row in rows
    ]


def csv_table(rows: list[AgentCallRow]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Agent Called", "Request", "Response", "Tools Called"])
    for row in rows:
        writer.writerow(
            [
                row.agent_name or "(unknown/direct)",
                row.request,
                row.response,
                ", ".join(row.tools) if row.tools else "",
            ]
        )
    return buffer.getvalue().strip()


def render_rows(rows: list[AgentCallRow], output_format: str) -> str:
    if output_format == "csv":
        return csv_table(rows)
    if output_format == "json":
        return json.dumps(rows_as_dicts(rows), ensure_ascii=False, indent=2)
    return markdown_table(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Get question/answer/tools-called table for an agent interaction from VS Code Copilot transcripts."
        )
    )
    parser.add_argument("--agent-name", required=True, help="Agent name label (for your reference)")
    parser.add_argument("--question", required=True, help="Question text")
    parser.add_argument(
        "--workspace-storage",
        type=Path,
        default=default_workspace_storage(),
        help="Path to VS Code workspaceStorage",
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=0,
        help="Wait for transcript update (useful right after asking the question)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Polling interval while waiting",
    )
    parser.add_argument(
        "--no-invoke",
        action="store_true",
        help="Do not invoke the agent via CLI; only read existing transcript data.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "csv", "json"),
        default="markdown",
        help="Output format for the extracted rows.",
    )
    args = parser.parse_args()

    invoke_prompt = ""
    invoke_error = ""
    invoke_start_ts = ""

    if not args.no_invoke:
        invoke_start_ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"
        invoke_prompt, invoke_error = invoke_agent_via_cli(args.agent_name, args.question)

    deadline = time.time() + max(args.wait_seconds, 0)

    while True:
        files = find_transcript_files(args.workspace_storage)
        events = parse_events(files)
        variants = [args.question]
        if invoke_prompt:
            variants.append(invoke_prompt)

        event_row = find_interaction(
            events,
            args.question,
            question_variants=variants,
            since_timestamp=invoke_start_ts if invoke_start_ts else None,
        )
        event_row.agent_name = args.agent_name
        rows = [event_row]

        _, chat_rows = find_chat_session_interaction(
            args.workspace_storage,
            args.question,
            question_variants=variants,
        )
        if chat_rows:
            for row in chat_rows:
                if not row.agent_name:
                    row.agent_name = args.agent_name
            rows = chat_rows

        found_complete_row = any("not found yet" not in row.response for row in rows)
        if found_complete_row or args.wait_seconds <= 0 or time.time() >= deadline:
            print(f"Agent: {args.agent_name}")
            print(render_rows(rows, args.format))

            if invoke_prompt:
                print(f"\nInvoke Prompt: {invoke_prompt}")
            if invoke_error:
                print(f"Invoke Status: {invoke_error}")
            elif not args.no_invoke:
                print("Invoke Status: sent via `code chat --mode agent --reuse-window`.")

            print("\nNote: The script invokes via VS Code CLI and reads transcript events for answer/tool extraction.")
            return 0

        time.sleep(max(args.poll_interval, 0.2))


if __name__ == "__main__":
    raise SystemExit(main())
