"""Capture and summarize Copilot agent traces from VS Code transcript files.

Example:
python agent_trace_table.py --agent-name "Web Q&A Agent" --question "where is New york" --wait-seconds 120
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Event:
    """Normalized event record loaded from transcript JSONL files."""

    event_type: str
    timestamp: str
    data: dict[str, Any]


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace so text matching is more tolerant."""

    return " ".join((text or "").strip().lower().split())


def default_workspace_storage() -> Path:
    """Return the default VS Code workspaceStorage directory on Windows."""

    appdata = os.environ.get("APPDATA")
    if not appdata:
        raise RuntimeError("APPDATA is not set")
    return Path(appdata) / "Code" / "User" / "workspaceStorage"


def resolve_code_cli() -> str | None:
    """Locate the VS Code CLI executable used to invoke agent chat."""

    for candidate in ("code", "code.cmd", "Code.exe"):
        hit = shutil.which(candidate)
        if hit:
            return hit

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

    for path in candidates:
        if path.exists():
            return str(path)
    return None


def find_transcript_files(workspace_storage: Path) -> list[Path]:
    """Find transcript JSONL files under Copilot chat storage locations."""

    files: list[Path] = []
    for root in workspace_storage.rglob("GitHub.copilot-chat"):
        transcript_dir = root / "transcripts"
        if transcript_dir.is_dir():
            files.extend(sorted(transcript_dir.glob("*.jsonl")))
    return sorted(files)


def parse_events(files: list[Path]) -> list[Event]:
    """Parse raw JSONL transcript files into sorted Event objects."""

    events: list[Event] = []
    for file_path in files:
        try:
            with file_path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
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

    events.sort(key=lambda item: item.timestamp)
    return events


def find_chat_session_files(workspace_storage: Path) -> list[Path]:
    """Find chat session files that can provide fallback question/answer metadata."""

    files: list[Path] = []
    for root in workspace_storage.rglob("chatSessions"):
        if root.is_dir():
            files.extend(sorted(root.glob("*.jsonl")))
    return sorted(files)


def extract_tool_name(value: dict[str, Any]) -> str:
    """Extract a tool name from multiple possible serialized response shapes."""

    tool_id = str(value.get("toolId", "")).strip()
    if tool_id:
        return tool_id

    tool_name = str(value.get("toolName", "")).strip()
    if tool_name:
        return tool_name

    return extract_text(value.get("invocationMessage"))


def merge_tools(tools: list[str]) -> list[str]:
    """Return tool names in first-seen order without duplicates."""

    deduped: list[str] = []
    seen: set[str] = set()
    for tool in tools:
        tool_name = str(tool).strip()
        if tool_name and tool_name not in seen:
            seen.add(tool_name)
            deduped.append(tool_name)
    return deduped


def parse_chat_session_file(file_path: Path) -> dict[int, dict[str, Any]]:
    """Parse a chat session file into request-indexed question/answer/tool records."""

    requests: dict[int, dict[str, Any]] = {}

    try:
        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not isinstance(obj, dict):
                    continue

                if obj.get("kind") == 0 and isinstance(obj.get("v"), dict):
                    session_id = str(obj["v"].get("sessionId", "")).strip()
                    if session_id:
                        requests.setdefault(-1, {})["session_id"] = session_id
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
                    {"question": "", "answer": "", "tools": [], "session_id": ""},
                )

                session_info = requests.get(-1, {})
                if isinstance(session_info, dict):
                    session_id = str(session_info.get("session_id", "")).strip()
                    if session_id and not record.get("session_id"):
                        record["session_id"] = session_id

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
                    for item in value:
                        if not isinstance(item, dict):
                            continue

                        if item.get("kind") == "toolInvocationSerialized":
                            tool_name = extract_tool_name(item)
                            if tool_name:
                                response_tools.append(tool_name)

                            tool_specific_data = item.get("toolSpecificData")
                            if isinstance(tool_specific_data, dict):
                                result = str(tool_specific_data.get("result", "")).strip()
                                if result and not record.get("answer"):
                                    record["answer"] = result
                            continue

                        text = extract_text(item)
                        if text:
                            response_texts.append(text)

                    if response_texts:
                        record["answer"] = response_texts[-1]
                    if response_tools:
                        record_tools = record.setdefault("tools", [])
                        if isinstance(record_tools, list):
                            record_tools.extend(response_tools)
    except OSError:
        return {}

    return requests


def find_chat_session_interaction(
    workspace_storage: Path,
    question: str,
    question_variants: list[str] | None = None,
) -> dict[str, Any] | None:
    """Find the best matching interaction in chatSessions for a given query."""

    q_norm = normalize(question)
    variant_norms = {q_norm}
    if question_variants:
        variant_norms.update(normalize(v) for v in question_variants if v)

    for file_path in find_chat_session_files(workspace_storage):
        requests = parse_chat_session_file(file_path)
        for index, record in requests.items():
            if index < 0 or not isinstance(record, dict):
                continue

            session_question = str(record.get("question", ""))
            session_norm = normalize(session_question)
            if not session_norm:
                continue

            if session_norm in variant_norms or any(v and v in session_norm for v in variant_norms) or (
                q_norm and q_norm in session_norm
            ):
                return {
                    "session_id": str(record.get("session_id", "")).strip(),
                    "question": session_question,
                    "answer": str(record.get("answer", "")).strip(),
                    "tools": merge_tools([str(tool) for tool in record.get("tools", [])]),
                }

    return None


def find_session_transcript_file(workspace_storage: Path, session_id: str) -> Path | None:
    """Locate the transcript file tied to a specific session id."""

    if not session_id:
        return None

    for file_path in workspace_storage.rglob(f"{session_id}.jsonl"):
        if file_path.parent.name == "transcripts":
            return file_path
    return None


def event_to_json_line(event: Event) -> str:
    """Serialize one Event to a single JSONL line."""

    return json.dumps(
        {
            "type": event.event_type,
            "timestamp": event.timestamp,
            "data": event.data,
        },
        ensure_ascii=False,
    )


def build_query_variants(agent_name: str, query: str) -> list[str]:
    """Build alternate query forms to improve transcript matching accuracy."""

    variants = [query]
    if agent_name:
        variants.append(f"@{agent_name} {query}")
    return [item for item in variants if item.strip()]


def extract_trace_window(
    events: list[Event],
    query_variants: list[str],
    since_timestamp: str,
) -> tuple[list[Event], bool, str]:
    """Extract the target user turn and subsequent assistant/tool events.

    Returns:
    - trace events for the matched turn,
    - whether a completed answer turn was detected,
    - the latest non-tool assistant answer text.
    """

    variant_norms = {normalize(item) for item in query_variants if item}

    user_index = -1
    contains_index = -1
    for idx in range(len(events) - 1, -1, -1):
        event = events[idx]
        if event.event_type != "user.message":
            continue
        if since_timestamp and event.timestamp < since_timestamp:
            continue

        content = normalize(str(event.data.get("content", "")))
        if not content:
            continue

        if content in variant_norms:
            user_index = idx
            break
        if contains_index == -1 and any(v and v in content for v in variant_norms):
            contains_index = idx

    if user_index == -1 and contains_index != -1:
        user_index = contains_index

    if user_index == -1:
        return [], False, ""

    trace: list[Event] = [events[user_index]]
    answer = ""
    answer_turn_complete = False

    for idx in range(user_index + 1, len(events)):
        event = events[idx]
        if event.event_type == "user.message":
            break
        trace.append(event)
        if event.event_type == "assistant.message":
            # Ignore interim status messages that include tool requests.
            tool_requests = event.data.get("toolRequests")
            has_tool_requests = isinstance(tool_requests, list) and len(tool_requests) > 0
            if not has_tool_requests:
                text = extract_text(event.data)
                if text:
                    answer = text
                    answer_turn_complete = False
        elif event.event_type == "assistant.turn_end" and answer:
            answer_turn_complete = True

    return trace, answer_turn_complete, answer


def extract_latest_completed_trace(events: list[Event], since_timestamp: str) -> tuple[list[Event], str]:
    """Fallback extractor for cases where exact query matching fails.

    Picks the most recent completed turn since `since_timestamp` and returns its
    trace plus the last non-tool assistant message content.
    """
    for end_idx in range(len(events) - 1, -1, -1):
        end_event = events[end_idx]
        if end_event.event_type != "assistant.turn_end":
            continue
        if since_timestamp and end_event.timestamp < since_timestamp:
            continue

        start_idx = end_idx
        while start_idx >= 0 and events[start_idx].event_type != "user.message":
            start_idx -= 1
        if start_idx < 0:
            continue

        trace = events[start_idx : end_idx + 1]
        answer = ""
        for event in trace:
            if event.event_type != "assistant.message":
                continue
            tool_requests = event.data.get("toolRequests")
            has_tool_requests = isinstance(tool_requests, list) and len(tool_requests) > 0
            if has_tool_requests:
                continue
            text = extract_text(event.data)
            if text:
                answer = text

        return trace, answer

    return [], ""


def extract_text(value: Any) -> str:
    """Recursively extract user-visible text from nested response payloads."""

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


def collect_tools(trace: list[Event]) -> list[str]:
    """Collect unique tool names invoked during the captured trace."""

    found: list[str] = []
    seen: set[str] = set()

    for event in trace:
        if event.event_type == "tool.execution_start":
            tool_name = str(event.data.get("toolName", "")).strip()
            if tool_name and tool_name not in seen:
                seen.add(tool_name)
                found.append(tool_name)

        if event.event_type == "assistant.message":
            tool_requests = event.data.get("toolRequests")
            if isinstance(tool_requests, list):
                for request in tool_requests:
                    if not isinstance(request, dict):
                        continue
                    tool_name = str(request.get("name", "")).strip()
                    if tool_name and tool_name not in seen:
                        seen.add(tool_name)
                        found.append(tool_name)

    return found


def slugify(value: str) -> str:
    """Create a filesystem-safe slug used in output bundle folder names."""

    out: list[str] = []
    for ch in value.strip():
        if ch.isalnum():
            out.append(ch.lower())
        elif ch in (" ", "-", "_"):
            out.append("_")
    slug = "".join(out).strip("_")
    return slug or "agent"


def write_trace_file(trace_file: Path, trace: list[Event]) -> None:
    """Write trace events to JSONL so each event is on its own line."""

    with trace_file.open("w", encoding="utf-8") as handle:
        for event in trace:
            handle.write(event_to_json_line(event))
            handle.write("\n")


def main() -> int:
    """Entry point: invoke agent chat, capture trace, and write summary artifacts."""

    parser = argparse.ArgumentParser(
        description="Invoke a Copilot agent and save a timestamped trace bundle while it runs."
    )
    parser.add_argument(
        "--agent-name",
        required=True,
        help="Agent name, e.g. Web Q&A Agent",
    )
    parser.add_argument(
        "--query",
        "--question",
        required=True,
        dest="query",
        help="Query to send to the agent",
    )
    parser.add_argument(
        "--workspace-storage",
        type=Path,
        default=default_workspace_storage(),
        help="Path to VS Code workspaceStorage",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "trace_output",
        help="Root folder where trace bundles are written",
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=120,
        help="Maximum time to wait for a complete answer",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds",
    )
    parser.add_argument(
        "--no-invoke",
        action="store_true",
        help="Do not invoke the agent; trace only from existing transcripts",
    )

    args = parser.parse_args()

    # Track run start in both UTC (for filtering events) and local timestamp (for folder names).
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    bundle_dir = args.output_root / "opentelemetry" / f"{slugify(args.agent_name)}_{stamp}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    trace_file = bundle_dir / "trace_events.jsonl"
    summary_file = bundle_dir / "summary.json"
    markdown_file = bundle_dir / "trace_summary.md"
    response_file = bundle_dir / "response.txt"

    prompt = f"@{args.agent_name} {args.query}" if args.agent_name else args.query
    invoke_status = "not requested"
    return_code: int | None = None
    stderr_text = ""

    # Start agent invocation unless caller requested trace-only mode.
    process: subprocess.Popen[str] | None = None
    if not args.no_invoke:
        cli = resolve_code_cli()
        if not cli:
            invoke_status = "VS Code CLI not found"
        else:
            cmd = [cli, "chat", "--mode", "agent", "--reuse-window", prompt]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            invoke_status = "started"

    # Poll transcripts until answer is complete, process exits, or timeout is reached.
    deadline = time.time() + max(args.wait_seconds, 0)
    best_trace: list[Event] = []
    final_answer = ""
    query_variants = build_query_variants(args.agent_name, args.query)
    session_context = find_chat_session_interaction(
        args.workspace_storage,
        args.query,
        query_variants,
    )

    while True:
        # Refresh both data sources every poll so newly created/updated files are seen.
        refreshed_session = find_chat_session_interaction(
            args.workspace_storage,
            args.query,
            query_variants,
        )
        if refreshed_session:
            session_context = refreshed_session

        transcript_files = find_transcript_files(args.workspace_storage)
        if session_context:
            session_id = str(session_context.get("session_id", "")).strip()
            session_transcript = find_session_transcript_file(args.workspace_storage, session_id)
            if session_transcript:
                transcript_files = [session_transcript, *transcript_files]

        # Preserve order while removing duplicates.
        transcript_files = list(dict.fromkeys(transcript_files))

        events = parse_events(transcript_files)
        trace, has_answer, answer = extract_trace_window(
            events,
            query_variants,
            started_utc,
        )

        # Fallback to most-recent completed turn if exact question matching misses.
        if not trace:
            latest_trace, latest_answer = extract_latest_completed_trace(events, started_utc)
            if latest_trace:
                trace = latest_trace
                has_answer = bool(latest_answer)
                if latest_answer:
                    answer = latest_answer

        if trace:
            best_trace = trace
            write_trace_file(trace_file, best_trace)
        if answer:
            final_answer = answer
        elif session_context:
            session_answer = str(session_context.get("answer", "")).strip()
            if session_answer:
                final_answer = session_answer

        process_done = True
        if process is not None:
            rc = process.poll()
            process_done = rc is not None
            if process_done and return_code is None:
                return_code = rc
                stderr_text = (process.stderr.read() if process.stderr else "").strip()

        if has_answer and (args.no_invoke or process_done):
            break
        if args.no_invoke and final_answer:
            break
        if time.time() >= deadline:
            break

        time.sleep(max(args.poll_interval, 0.2))

    # If no raw trace was found but session data exists, synthesize a minimal trace.
    if not best_trace and session_context:
        session_tools = [str(tool) for tool in session_context.get("tools", []) if str(tool).strip()]
        synthetic_trace = [
            Event(
                event_type="user.message",
                timestamp=started_utc,
                data={"content": prompt, "attachments": []},
            ),
        ]
        if session_tools:
            synthetic_trace.append(
                Event(
                    event_type="assistant.message",
                    timestamp=started_utc,
                    data={
                        "content": final_answer or str(session_context.get("answer", "")).strip(),
                        "toolRequests": [
                            {"name": tool_name, "toolCallId": tool_name, "arguments": "", "type": "function"}
                            for tool_name in session_tools
                        ],
                    },
                )
            )
            for tool_name in session_tools:
                synthetic_trace.append(
                    Event(
                        event_type="tool.execution_start",
                        timestamp=started_utc,
                        data={"toolName": tool_name},
                    )
                )
        synthetic_trace.append(
            Event(
                event_type="assistant.turn_end",
                timestamp=started_utc,
                data={"turnId": "0"},
            )
        )
        best_trace = synthetic_trace
        if not final_answer:
            final_answer = str(session_context.get("answer", "")).strip()
        write_trace_file(trace_file, best_trace)

    tools = collect_tools(best_trace)
    if session_context and not tools:
        tools = [str(tool) for tool in session_context.get("tools", []) if str(tool).strip()]

    if process is not None and return_code is None:
        rc = process.poll()
        if rc is not None:
            return_code = rc
            stderr_text = (process.stderr.read() if process.stderr else "").strip()

    if process is not None and return_code is not None:
        if return_code == 0:
            invoke_status = "finished"
        else:
            invoke_status = f"failed ({return_code})"

    # Persist machine-readable run summary.
    summary = {
        "agent_name": args.agent_name,
        "query": args.query,
        "prompt": prompt,
        "started_utc": started_utc,
        "bundle_dir": str(bundle_dir),
        "trace_file": str(trace_file),
        "response_file": str(response_file),
        "events_captured": len(best_trace),
        "tools_called": tools,
        "answer_found": bool(final_answer),
        "answer": final_answer or "(answer not found yet)",
        "invoke_status": invoke_status,
        "invoke_stderr": stderr_text,
    }

    with summary_file.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    with response_file.open("w", encoding="utf-8") as handle:
        handle.write((final_answer or "(answer not found yet)").strip() + "\n")

    # Persist a human-friendly markdown summary for quick inspection.
    lines = [
        f"# Agent Trace Summary",
        "",
        f"- Agent: {args.agent_name}",
        f"- Query: {args.query}",
        f"- Started (UTC): {started_utc}",
        f"- Invoke status: {invoke_status}",
        f"- Events captured: {len(best_trace)}",
        f"- Tools called: {', '.join(tools) if tools else '(none detected)'}",
        "",
        "## Answer",
        "",
        final_answer or "(answer not found yet)",
        "",
        "## Files",
        "",
        f"- {trace_file.name}",
        f"- {summary_file.name}",
        f"- {response_file.name}",
    ]

    with markdown_file.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).strip() + "\n")

    print(f"Trace bundle: {bundle_dir}")
    print(f"Trace file: {trace_file}")
    print(f"Summary: {summary_file}")
    print(f"Response: {response_file}")
    print(f"Tools called: {', '.join(tools) if tools else '(none detected)'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
