"""Batch-invoke a Copilot agent for every question in a test-data CSV and
write a results CSV with the real answer, tools called, and evaluation scores.

Example:

    python agent_trace_table_3.py --test-data test_data.csv  # uses Web Q&A Agent by default
    python agent_trace_table_3.py --test-data test_data.csv --no-eval  # skip evaluation

Evaluation metrics (requires: pip install deepeval):
    relevance_score          – answer relevance to the question
    groundedness_score       – answer grounded in retrieved context
    hallucination            – proportion of answer not grounded in context
    tool_selection_correctness – precision of tools selected for the task
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Optional DeepEval dependency — install with: pip install deepeval
try:
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval, HallucinationMetric
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    DEEPEVAL_AVAILABLE = True
except ImportError:  # pragma: no cover
    DEEPEVAL_AVAILABLE = False

# Optional sentence-transformers for semantic similarity — install with: pip install sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    np = None  # type: ignore[assignment]

# Optional sklearn for TF-IDF cosine similarity fallback — install with: pip install scikit-learn
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    SKLEARN_AVAILABLE = False

# Optional NLTK for BLEU score — install with: pip install nltk
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_BLEU_AVAILABLE = True
except ImportError:  # pragma: no cover
    NLTK_BLEU_AVAILABLE = False

# Optional rouge-score for ROUGE metrics — install with: pip install rouge-score
try:
    from rouge_score import rouge_scorer as rouge_scorer_lib
    ROUGE_AVAILABLE = True
except ImportError:  # pragma: no cover
    ROUGE_AVAILABLE = False


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
                                full_text = "\n".join(parts)
                                # Extract only the <userRequest> content to avoid
                                # the length filter blocking matches when VS Code
                                # wraps the question in a full context block.
                                user_req_match = re.search(
                                    r"<userRequest>\s*(.*?)\s*</userRequest>",
                                    full_text,
                                    re.DOTALL,
                                )
                                record["question"] = (
                                    user_req_match.group(1).strip()
                                    if user_req_match
                                    else full_text
                                )

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

    # Compute a generous length ceiling.  Session messages that are much longer
    # than the query are very unlikely to be the genuine agent turn we want —
    # they are almost always large context/attachment messages (e.g. ones that
    # embed an entire CSV file) that merely contain the query as a substring.
    # Capping at 3× the longest variant + 200 chars allows for agent-mode
    # prefixes ("@AgentName …") while filtering out bulk messages.
    max_session_len = max((len(v) for v in variant_norms), default=len(q_norm)) * 3 + 200

    for file_path in find_chat_session_files(workspace_storage):
        requests = parse_chat_session_file(file_path)
        for index, record in requests.items():
            if index < 0 or not isinstance(record, dict):
                continue

            session_question = str(record.get("question", ""))
            session_norm = normalize(session_question)
            if not session_norm:
                continue

            # Reject session messages that are far longer than the query to
            # avoid false-positive matches against bulk context messages.
            # Apply the length ceiling only for exact/variant matches, not
            # for substring checks where the session message is expected to
            # be longer than the query (e.g. VS Code context-wrapped messages).
            exact_match = session_norm in variant_norms or any(
                v and v in session_norm for v in variant_norms
            )
            substring_match = q_norm and q_norm in session_norm

            if exact_match and len(session_norm) > max_session_len:
                continue

            if exact_match or substring_match:
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

    # Same length ceiling used in find_chat_session_interaction: reject user
    # messages that are far longer than the query so we don't accidentally
    # match bulk context messages (e.g. ones that embed a whole CSV).
    max_content_len = max((len(v) for v in variant_norms), default=0) * 3 + 200

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
        if contains_index == -1 and len(content) <= max_content_len and any(v and v in content for v in variant_norms):
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
            # Ignore interim status messages that include tool requests,
            # EXCEPT when the only tool called is vscode_askQuestions — that
            # means the agent is asking for clarification and the message
            # content IS the answer.
            tool_requests = event.data.get("toolRequests")
            has_tool_requests = isinstance(tool_requests, list) and len(tool_requests) > 0
            only_ask_questions = has_tool_requests and all(
                isinstance(tr, dict) and tr.get("name") == "vscode_askQuestions"
                for tr in tool_requests
            )
            if not has_tool_requests or only_ask_questions:
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
            if events[start_idx].event_type == "session.start" and start_idx < end_idx:
                # Outer sessions invoked via CLI have no user.message; treat
                # session.start as the beginning of the turn.
                break
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
            only_ask_questions = has_tool_requests and all(
                isinstance(tr, dict) and tr.get("name") == "vscode_askQuestions"
                for tr in tool_requests
            )
            if has_tool_requests and not only_ask_questions:
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


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------


class EvaluationPipeline:
    """Modular evaluation pipeline wrapping DeepEval metrics with heuristic fallbacks.

    Instantiate once and call :meth:`evaluate_row` for every result record.
    Each metric degrades gracefully when DeepEval is not installed or when the
    required LLM credentials are unavailable.
    """

    def __init__(self, use_deepeval: bool = True, model: str = "gpt-4o-mini") -> None:
        self.use_deepeval = use_deepeval and DEEPEVAL_AVAILABLE
        self.model = model
        self._relevance_metric: Any = None
        self._groundedness_metric: Any = None
        self._hallucination_metric: Any = None
        self._st_model: Any = None
        if self.use_deepeval:
            self._init_metrics()
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self._st_model = None

    def _init_metrics(self) -> None:
        self._relevance_metric = AnswerRelevancyMetric(
            model=self.model,
            threshold=0.5,
            verbose_mode=False,
        )
        self._groundedness_metric = FaithfulnessMetric(
            model=self.model,
            threshold=0.5,
            verbose_mode=False,
        )
        self._hallucination_metric = HallucinationMetric(
            model=self.model,
            threshold=0.5,
            verbose_mode=False,
        )

    # ------------------------------------------------------------------
    # Individual metric methods
    # ------------------------------------------------------------------

    def compute_relevance(self, question: str, real_answer: str) -> float:
        """Score how relevant *real_answer* is to *question* (0–1)."""
        if self.use_deepeval and self._relevance_metric is not None:
            try:
                tc = LLMTestCase(input=question, actual_output=real_answer)
                self._relevance_metric.measure(tc)
                score = self._relevance_metric.score
                if score is not None:
                    return float(score)
            except Exception:
                pass
        return self._word_overlap(question, real_answer)

    def compute_groundedness(
        self,
        question: str,
        real_answer: str,
        context: list[str] | None = None,
    ) -> float:
        """Score whether *real_answer* is grounded in retrieved context (0–1).

        Falls back to a word-count heuristic when no retrieval context is
        provided (typical for direct agent invocations).
        """
        if self.use_deepeval and self._groundedness_metric is not None and context:
            try:
                tc = LLMTestCase(
                    input=question,
                    actual_output=real_answer,
                    retrieval_context=context,
                )
                self._groundedness_metric.measure(tc)
                score = self._groundedness_metric.score
                if score is not None:
                    return float(score)
            except Exception:
                pass
        if not real_answer or real_answer == "(answer not captured)":
            return 0.0
        return min(1.0, len(real_answer.split()) / 50.0)

    def compute_hallucination(
        self,
        question: str,
        real_answer: str,
        context: list[str] | None = None,
    ) -> float:
        """Score hallucination in *real_answer* (0 = none, 1 = fully hallucinated).

        Uses DeepEval HallucinationMetric when context is available; otherwise
        falls back to ``1 - groundedness`` as a heuristic.
        """
        if self.use_deepeval and self._hallucination_metric is not None and context:
            try:
                tc = LLMTestCase(
                    input=question,
                    actual_output=real_answer,
                    context=context,
                )
                self._hallucination_metric.measure(tc)
                score = self._hallucination_metric.score
                if score is not None:
                    return float(score)
            except Exception:
                pass
        groundedness = self.compute_groundedness(question, real_answer, context)
        return round(1.0 - groundedness, 4)

    def compute_tool_selection_correctness(
        self,
        tools_called: str,
        expected_tools: list[str] | None = None,
    ) -> float:
        """Score precision of tool selection (0–1).

        Measures what fraction of the tools actually called were appropriate
        (present in *expected_tools*).  Without expected_tools a binary
        heuristic is applied: tools used → 1.0, none used → 0.5.
        """
        actual = {
            t.strip()
            for t in (tools_called or "").split(";")
            if t.strip() and t.strip() != "(none)"
        }
        if expected_tools is not None:
            exp = {t.strip() for t in expected_tools if t.strip()}
            if not exp and not actual:
                return 1.0
            if not actual:
                return 0.0
            if not exp:
                return 0.5
            # Precision: fraction of called tools that are in the expected set
            return len(exp & actual) / len(actual)
        return 1.0 if actual else 0.5

    def compute_tool_usage_correctness(
        self,
        tools_called: str,
        expected_tools: list[str] | None = None,
    ) -> float:
        """Score whether the appropriate tools were invoked (0–1).

        When *expected_tools* are provided the score is the Jaccard similarity
        between expected and actual tool sets.  Without expected tools a simple
        binary heuristic is applied (tools used → 1.0, none → 0.5).
        """
        actual = {
            t.strip()
            for t in (tools_called or "").split(";")
            if t.strip() and t.strip() != "(none)"
        }
        if expected_tools is not None:
            exp = {t.strip() for t in expected_tools if t.strip()}
            if not exp:
                return 1.0 if not actual else 0.5
            return len(exp & actual) / len(exp | actual)
        return 1.0 if actual else 0.5

    # ------------------------------------------------------------------
    # Aggregate evaluation
    # ------------------------------------------------------------------

    def compute_semantic_similarity(self, expected: str, actual: str) -> float:
        """Compute semantic similarity between *expected* and *actual* answer (0–1).

        Uses sentence-transformers cosine similarity when available, falls back
        to TF-IDF cosine similarity (sklearn), and finally to word-overlap.
        """
        if not expected or not actual or actual == "(answer not captured)":
            return 0.0

        if self._st_model is not None and np is not None:
            try:
                emb = self._st_model.encode([expected, actual])
                norm_a = np.linalg.norm(emb[0])
                norm_b = np.linalg.norm(emb[1])
                if norm_a > 0 and norm_b > 0:
                    sim = float(np.dot(emb[0], emb[1]) / (norm_a * norm_b))
                    return max(0.0, min(1.0, sim))
            except Exception:
                pass

        if SKLEARN_AVAILABLE:
            try:
                vect = TfidfVectorizer()
                tfidf = vect.fit_transform([expected, actual])
                sim = float(sklearn_cosine_similarity(tfidf[0], tfidf[1])[0][0])
                return max(0.0, min(1.0, sim))
            except Exception:
                pass

        return self._word_overlap(expected, actual)

    def compute_bleu_rouge(self, expected: str, actual: str) -> float:
        """Compute a combined BLEU / ROUGE-L score between *expected* and *actual* (0–1).

        Averages BLEU-1 (nltk) and ROUGE-L F1 (rouge-score) when both libraries
        are available.  Falls back to whichever is available, or to simple
        word-overlap when neither is installed.
        """
        if not expected or not actual or actual == "(answer not captured)":
            return 0.0

        scores: list[float] = []

        if NLTK_BLEU_AVAILABLE:
            try:
                ref = [expected.lower().split()]
                hyp = actual.lower().split()
                sf = SmoothingFunction().method1
                bleu = float(sentence_bleu(ref, hyp, smoothing_function=sf))
                scores.append(max(0.0, min(1.0, bleu)))
            except Exception:
                pass

        if ROUGE_AVAILABLE:
            try:
                scorer = rouge_scorer_lib.RougeScorer(["rougeL"], use_stemmer=True)
                result = scorer.score(expected, actual)
                rouge_l = float(result["rougeL"].fmeasure)
                scores.append(max(0.0, min(1.0, rouge_l)))
            except Exception:
                pass

        if scores:
            return round(sum(scores) / len(scores), 4)
        return self._word_overlap(expected, actual)

    def evaluate_row(
        self,
        question: str,
        expected_answer: str,
        real_answer: str,
        tools_called: str,
        context: list[str] | None = None,
        expected_tools: list[str] | None = None,
    ) -> dict[str, Any]:
        """Evaluate one result row and return the fully augmented record."""
        relevance = self.compute_relevance(question, real_answer)
        groundedness = self.compute_groundedness(question, real_answer, context)
        hallucination = self.compute_hallucination(question, real_answer, context)
        tool_sel = self.compute_tool_selection_correctness(tools_called, expected_tools)
        semantic_sim = self.compute_semantic_similarity(expected_answer, real_answer)
        bleu_rouge = self.compute_bleu_rouge(expected_answer, real_answer)
        avg_score = (relevance + groundedness + (1.0 - hallucination) + tool_sel + semantic_sim + bleu_rouge) / 6.0

        # Scale all 0-1 scores to the 1-100 range
        relevance_scaled = self._to_100_scale(relevance)
        groundedness_scaled = self._to_100_scale(groundedness)
        hallucination_scaled = self._to_100_scale(hallucination)
        tool_sel_scaled = self._to_100_scale(tool_sel)
        semantic_sim_scaled = self._to_100_scale(semantic_sim)
        bleu_rouge_scaled = self._to_100_scale(bleu_rouge)
        avg_score_scaled = self._to_100_scale(avg_score)

        eval_summary = json.dumps(
            {
                "relevance": relevance_scaled,
                "groundedness": groundedness_scaled,
                "hallucination": hallucination_scaled,
                "tool_selection_correctness": tool_sel_scaled,
                "semantic_similarity": semantic_sim_scaled,
                "bleu_rouge_score": bleu_rouge_scaled,
                "avg_score": avg_score_scaled,
                "evaluator": "deepeval" if self.use_deepeval else "heuristic",
            },
            ensure_ascii=False,
        )

        return {
            "question": question,
            "expected_answer": expected_answer,
            "real_answer": real_answer,
            "tools_called": tools_called,
            "relevance_score": relevance_scaled,
            "groundedness_score": groundedness_scaled,
            "hallucination": hallucination_scaled,
            "tool_selection_correctness": tool_sel_scaled,
            "semantic_similarity": semantic_sim_scaled,
            "bleu_rouge_score": bleu_rouge_scaled,
            "eval_result": eval_summary,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_100_scale(score: float) -> float:
        """Map a 0–1 score to the 1–100 scale (0 → 1, 1 → 100)."""
        return round(1.0 + float(score) * 99.0, 2)

    @staticmethod
    def _word_overlap(reference: str, hypothesis: str) -> float:
        """Token-level recall: fraction of *reference* words found in *hypothesis*."""
        ref_tokens = set((reference or "").lower().split())
        hyp_tokens = set((hypothesis or "").lower().split())
        if not ref_tokens:
            return 0.0
        return len(ref_tokens & hyp_tokens) / len(ref_tokens)


# ---------------------------------------------------------------------------
# Per-question invocation
# ---------------------------------------------------------------------------

def invoke_and_capture(
    agent_name: str,
    question: str,
    workspace_storage: Path,
    wait_seconds: int,
    poll_interval: float,
    cli: str | None,
) -> tuple[str, list[str]]:
    """Invoke the agent for one question and return (real_answer, tools_called)."""

    started_utc = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"
    prompt = f"@{agent_name} {question}" if agent_name else question
    query_variants = build_query_variants(agent_name, question)

    process: subprocess.Popen[str] | None = None
    if cli:
        cmd = [cli, "chat", "--mode", "agent", "--reuse-window", prompt]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    deadline = time.time() + max(wait_seconds, 0)
    best_trace: list[Event] = []
    final_answer = ""
    session_context: dict[str, Any] | None = None

    while True:
        # Refresh session data every poll
        refreshed = find_chat_session_interaction(workspace_storage, question, query_variants)
        if refreshed:
            session_context = refreshed

        transcript_files = find_transcript_files(workspace_storage)
        if session_context:
            sid = str(session_context.get("session_id", "")).strip()
            session_tf = find_session_transcript_file(workspace_storage, sid)
            if session_tf:
                transcript_files = [session_tf, *transcript_files]
        transcript_files = list(dict.fromkeys(transcript_files))

        events = parse_events(transcript_files)
        trace, has_answer, answer = extract_trace_window(events, query_variants, started_utc)

        if not trace:
            latest_trace, latest_answer = extract_latest_completed_trace(events, started_utc)
            if latest_trace:
                trace = latest_trace
                has_answer = bool(latest_answer)
                if latest_answer:
                    answer = latest_answer

        if trace:
            best_trace = trace
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

        if has_answer and process_done:
            break
        if final_answer and not process:
            break
        if time.time() >= deadline:
            break

        time.sleep(max(poll_interval, 0.2))

    # Synthesize minimal trace from session data if transcript capture missed
    if not best_trace and session_context:
        session_tools = [str(t) for t in session_context.get("tools", []) if str(t).strip()]
        synthetic: list[Event] = [
            Event(
                event_type="user.message",
                timestamp=started_utc,
                data={"content": prompt, "attachments": []},
            )
        ]
        if session_tools:
            synthetic.append(Event(
                event_type="assistant.message",
                timestamp=started_utc,
                data={
                    "content": final_answer or str(session_context.get("answer", "")).strip(),
                    "toolRequests": [
                        {"name": t, "toolCallId": t, "arguments": "", "type": "function"}
                        for t in session_tools
                    ],
                },
            ))
            for t in session_tools:
                synthetic.append(Event(
                    event_type="tool.execution_start",
                    timestamp=started_utc,
                    data={"toolName": t},
                ))
        synthetic.append(Event(
            event_type="assistant.turn_end",
            timestamp=started_utc,
            data={"turnId": "0"},
        ))
        best_trace = synthetic
        if not final_answer:
            final_answer = str(session_context.get("answer", "")).strip()

    tools = collect_tools(best_trace)
    if session_context and not tools:
        tools = [str(t) for t in session_context.get("tools", []) if str(t).strip()]

    return (
        final_answer or "(answer not captured)",
        tools,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read questions from a test-data CSV, invoke a Copilot agent for each one, "
            "and write a results CSV with question, expected_answer, real_answer, tools_called."
        )
    )
    parser.add_argument(
        "--test-data",
        required=True,
        type=Path,
        help="Path to the input CSV (must have 'question' and 'expected_answer' columns).",
    )
    parser.add_argument(
        "--agent-name",
        default="Web Q&A Agent",
        help="Copilot agent name (default: 'Web Q&A Agent').",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory where the results CSV is written (default: script directory).",
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=120,
        help="Maximum seconds to wait per question for a complete answer (default: 120).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--workspace-storage",
        type=Path,
        default=default_workspace_storage(),
        help="Path to VS Code workspaceStorage.",
    )
    parser.add_argument(
        "--no-invoke",
        action="store_true",
        help="Skip agent invocation; read answers from existing transcripts only.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluation metrics; output only the base trace columns.",
    )
    parser.add_argument(
        "--eval-model",
        default="gpt-4o-mini",
        help="LLM model name passed to DeepEval metrics (default: 'gpt-4o-mini').",
    )
    args = parser.parse_args()

    # Validate input CSV
    if not args.test_data.exists():
        print(f"ERROR: test-data file not found: {args.test_data}", file=sys.stderr)
        return 1

    rows: list[dict[str, str]] = []
    with args.test_data.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if (
            reader.fieldnames is None
            or "question" not in reader.fieldnames
            or "expected_answer" not in reader.fieldnames
        ):
            print(
                "ERROR: CSV must contain 'question' and 'expected_answer' columns.",
                file=sys.stderr,
            )
            return 1
        for row in reader:
            rows.append({
                "question": row["question"].strip(),
                "expected_answer": row["expected_answer"].strip(),
            })

    if not rows:
        print("ERROR: No rows found in test-data CSV.", file=sys.stderr)
        return 1

    # Resolve CLI once for the whole run
    cli: str | None = None
    if not args.no_invoke:
        cli = resolve_code_cli()
        if not cli:
            print("WARNING: VS Code CLI not found. Running in no-invoke mode.", file=sys.stderr)

    # Initialise the evaluation pipeline (None when --no-eval is passed)
    evaluator: EvaluationPipeline | None = None
    if not args.no_eval:
        evaluator = EvaluationPipeline(
            use_deepeval=DEEPEVAL_AVAILABLE,
            model=args.eval_model,
        )

    # Output file: <slug>_YYYYMMDD_HHMMSS_results.csv
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_filename = f"{slugify(args.agent_name)}_{stamp}_results.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / out_filename

    total = len(rows)
    print(f"Agent   : {args.agent_name}")
    print(f"Input   : {args.test_data} ({total} questions)")
    print(f"Output  : {out_path}")
    print(f"Timeout : {args.wait_seconds}s per question")
    if evaluator is not None:
        eval_backend = "DeepEval" if DEEPEVAL_AVAILABLE else "heuristic"
        print(f"Eval    : {eval_backend} ({args.eval_model})")
    else:
        print("Eval    : disabled (--no-eval)")
    print()

    _FIELDNAMES = [
        "question",
        "expected_answer",
        "real_answer",
        "tools_called",
        "relevance_score",
        "groundedness_score",
        "hallucination",
        "tool_selection_correctness",
        "semantic_similarity",
        "bleu_rouge_score",
        "eval_result",
    ]

    results: list[dict[str, Any]] = []

    for i, row in enumerate(rows, 1):
        question = row["question"]
        expected = row["expected_answer"]
        row_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        print(f"[{i}/{total}] {question[:80]}")

        real_answer, tools = invoke_and_capture(
            agent_name=args.agent_name,
            question=question,
            workspace_storage=args.workspace_storage,
            wait_seconds=args.wait_seconds,
            poll_interval=args.poll_interval,
            cli=cli,
        )

        tools_str = "; ".join(tools) if tools else "(none)"
        print(f"        Answer : {real_answer[:120]}")
        print(f"        Tools  : {tools_str}")

        if evaluator is not None:
            record = evaluator.evaluate_row(
                question=question,
                expected_answer=expected,
                real_answer=real_answer,
                tools_called=tools_str,
            )
            print(
                f"        Scores : relevance={record['relevance_score']}"
                f"  groundedness={record['groundedness_score']}"
                f"  hallucination={record['hallucination']}"
                f"  tool_selection={record['tool_selection_correctness']}"
                f"  semantic_sim={record['semantic_similarity']}"
                f"  bleu_rouge={record['bleu_rouge_score']}"
            )
        else:
            record = {
                "question": question,
                "expected_answer": expected,
                "real_answer": real_answer,
                "tools_called": tools_str,
                "relevance_score": "",
                "groundedness_score": "",
                "hallucination": "",
                "tool_selection_correctness": "",
                "semantic_similarity": "",
                "bleu_rouge_score": "",
                "eval_result": "",
            }
        print()

        results.append(record)

        # Write incrementally so partial results survive an early exit
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
            writer.writeheader()
            writer.writerows(results)

    print(f"Done. Results written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
