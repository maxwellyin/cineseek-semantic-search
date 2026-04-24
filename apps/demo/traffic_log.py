from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
import sqlite3

from fastapi import Request


APP_DIR = Path(__file__).resolve().parent
TRAFFIC_DB_PATH = APP_DIR / "traffic.sqlite3"
RETENTION_DAYS = 7

TRACKED_PAGE_PATHS = {"/", "/home", "/search", "/demo", "/demo/input"}
SEARCH_PATHS = {"/search/results", "/demo/outcome"}
IGNORED_PATH_PREFIXES = ("/static/",)
IGNORED_PATHS = {
    "/favicon.ico",
    "/apple-touch-icon.png",
    "/apple-touch-icon-precomposed.png",
    "/health",
    "/ops/traffic",
}


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(TRAFFIC_DB_PATH, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def ensure_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS traffic_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                path TEXT NOT NULL,
                query_text TEXT,
                use_agent INTEGER,
                status_code INTEGER,
                client_ip TEXT,
                user_agent TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_traffic_events_created_at ON traffic_events(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_traffic_events_type ON traffic_events(event_type)")


def cleanup_old_events() -> None:
    cutoff = (datetime.now(UTC) - timedelta(days=RETENTION_DAYS)).isoformat()
    with _connect() as conn:
        conn.execute("DELETE FROM traffic_events WHERE created_at < ?", (cutoff,))


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    if forwarded:
        return forwarded
    return request.client.host if request.client else ""


def should_track(request: Request) -> bool:
    if request.method != "GET":
        return False
    path = request.url.path
    if path in IGNORED_PATHS:
        return False
    if path.startswith(IGNORED_PATH_PREFIXES):
        return False
    return path in TRACKED_PAGE_PATHS or path in SEARCH_PATHS


def record_request(request: Request, status_code: int) -> None:
    if not should_track(request):
        return

    ensure_db()
    cleanup_old_events()

    path = request.url.path
    event_type = "search" if path in SEARCH_PATHS else "page_view"
    query_text = request.query_params.get("text") if event_type == "search" else None
    use_agent_value = request.query_params.get("use_agent")
    use_agent = None
    if use_agent_value is not None:
        use_agent = 1 if use_agent_value == "1" else 0

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO traffic_events (
                created_at, event_type, path, query_text, use_agent, status_code, client_ip, user_agent
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(UTC).isoformat(),
                event_type,
                path,
                query_text,
                use_agent,
                status_code,
                _client_ip(request),
                request.headers.get("user-agent", "")[:300],
            ),
        )


def fetch_dashboard(limit: int = 100) -> dict[str, object]:
    ensure_db()
    cleanup_old_events()
    cutoff = (datetime.now(UTC) - timedelta(days=RETENTION_DAYS)).isoformat()
    day_cutoff = (datetime.now(UTC) - timedelta(days=1)).isoformat()

    with _connect() as conn:
        total_last_week = conn.execute(
            """
            SELECT COUNT(DISTINCT client_ip)
            FROM traffic_events
            WHERE created_at >= ? AND event_type = 'page_view' AND client_ip IS NOT NULL AND client_ip != ''
            """,
            (cutoff,),
        ).fetchone()[0]
        total_last_day = conn.execute(
            """
            SELECT COUNT(DISTINCT client_ip)
            FROM traffic_events
            WHERE created_at >= ? AND event_type = 'page_view' AND client_ip IS NOT NULL AND client_ip != ''
            """,
            (day_cutoff,),
        ).fetchone()[0]
        search_last_week = conn.execute(
            "SELECT COUNT(*) FROM traffic_events WHERE created_at >= ? AND event_type = 'search'",
            (cutoff,),
        ).fetchone()[0]
        search_last_day = conn.execute(
            "SELECT COUNT(*) FROM traffic_events WHERE created_at >= ? AND event_type = 'search'",
            (day_cutoff,),
        ).fetchone()[0]
        rows = conn.execute(
            """
            SELECT created_at, event_type, path, query_text, use_agent, status_code, client_ip, user_agent
            FROM traffic_events
            WHERE created_at >= ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (cutoff, limit),
        ).fetchall()
        top_queries_raw = conn.execute(
            """
            SELECT query_text
            FROM traffic_events
            WHERE created_at >= ? AND event_type = 'search' AND query_text IS NOT NULL AND query_text != ''
            """,
            (cutoff,),
        ).fetchall()

    top_queries = Counter(row["query_text"] for row in top_queries_raw).most_common(12)
    recent_events = [dict(row) for row in rows]
    return {
        "summary": {
            "last_24h_visits": int(total_last_day),
            "last_7d_visits": int(total_last_week),
            "last_24h_searches": int(search_last_day),
            "last_7d_searches": int(search_last_week),
        },
        "top_queries": [{"query": query, "count": count} for query, count in top_queries],
        "recent_events": recent_events,
    }
