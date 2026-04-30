from __future__ import annotations

import hmac
import os
from pathlib import Path
import re
from urllib.parse import quote_plus, urlencode
from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape
from starlette.concurrency import run_in_threadpool

try:
    from . import network
    from . import search_mcp_server
    from . import traffic_log
except ImportError:
    import network
    import search_mcp_server
    import traffic_log


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent.parent
FRONTEND_DIR = REPO_ROOT / "frontend"
FRONTEND_DIST_DIR = FRONTEND_DIR / "dist"
FRONTEND_INDEX_HTML = FRONTEND_DIST_DIR / "index.html"
FRONTEND_PUBLIC_DIR = FRONTEND_DIR / "public"
DEFAULT_HOME_QUERY = "Mind-bending movies like Inception but darker"
PUBLIC_MCP_PREFIX = "/mcp/search"
PUBLIC_MCP_BEARER_TOKEN = os.environ.get("FLCR_PUBLIC_MCP_BEARER_TOKEN", "").strip()
INTERNAL_MCP_SERVER_URL = os.environ.get("FLCR_INTERNAL_MCP_SERVER_URL", "").strip()


@asynccontextmanager
async def app_lifespan(app_instance: FastAPI):
    traffic_log.ensure_db()
    async with search_mcp_server.mcp_app.lifespan(app_instance):
        yield


app = FastAPI(lifespan=app_lifespan)

# Allow CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=str(APP_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
app.mount("/agent-tools", search_mcp_server.mcp_app)
app.mount(PUBLIC_MCP_PREFIX, search_mcp_server.mcp_app)
if (FRONTEND_DIST_DIR / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST_DIR / "assets")), name="frontend-assets")


def render_inline_markdown(value: str | None) -> Markup:
    escaped = escape(value or "")
    html = str(escaped)
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", html)
    html = re.sub(r"`(.+?)`", r"<code>\1</code>", html)
    return Markup(html)


templates.env.filters["inline_markdown"] = render_inline_markdown
templates.env.filters["query_param"] = lambda value: quote_plus(str(value or ""))


def render_template(request: Request, template_name: str, **context):
    return templates.TemplateResponse(request, template_name, context)


def frontend_available() -> bool:
    return FRONTEND_INDEX_HTML.is_file()


def render_frontend() -> FileResponse:
    return FileResponse(FRONTEND_INDEX_HTML)


@app.middleware("http")
async def mcp_auth_middleware(request: Request, call_next):
    if request.url.path.startswith(PUBLIC_MCP_PREFIX):
        auth_header = request.headers.get("authorization", "").strip()
        expected = f"Bearer {PUBLIC_MCP_BEARER_TOKEN}" if PUBLIC_MCP_BEARER_TOKEN else ""
        if not expected or not hmac.compare_digest(auth_header, expected):
            return JSONResponse(
                {
                    "error": "Unauthorized",
                    "detail": "Provide a valid Bearer token to access the public MCP endpoint.",
                },
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
    return await call_next(request)


@app.middleware("http")
async def traffic_middleware(request: Request, call_next):
    response = await call_next(request)
    await run_in_threadpool(traffic_log.record_request, request, response.status_code)
    return response


@app.get("/", response_class=HTMLResponse, name="home")
@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    if frontend_available():
        return render_frontend()
    return render_template(request, "home.html")


@app.head("/")
async def head_home():
    return Response(status_code=200)


@app.get("/favicon.svg")
async def frontend_favicon():
    frontend_favicon_path = FRONTEND_PUBLIC_DIR / "favicon.svg"
    if frontend_favicon_path.is_file():
        return FileResponse(frontend_favicon_path)
    return RedirectResponse(url="/static/favicon.svg", status_code=308)


@app.get("/icons.svg")
async def frontend_icons():
    frontend_icons_path = FRONTEND_PUBLIC_DIR / "icons.svg"
    if frontend_icons_path.is_file():
        return FileResponse(frontend_icons_path)
    return Response(status_code=404)


@app.get("/favicon.ico")
@app.get("/apple-touch-icon.png")
@app.get("/apple-touch-icon-precomposed.png")
async def favicon_alias():
    return RedirectResponse(url="/favicon.svg", status_code=308)


@app.get("/health")
async def health():
    status = await run_in_threadpool(network.health_status)
    status_code = 200 if status.get("status") == "ok" else 503
    return JSONResponse(status, status_code=status_code)


@app.get("/ops/traffic", response_class=HTMLResponse)
async def traffic_dashboard(request: Request):
    dashboard = await run_in_threadpool(traffic_log.fetch_dashboard)
    return render_template(
        request,
        "traffic.j2",
        summary=dashboard["summary"],
        top_queries=dashboard["top_queries"],
        recent_events=dashboard["recent_events"],
    )


@app.get("/search", response_class=HTMLResponse, name="search")
@app.get("/demo", response_class=HTMLResponse)
@app.get("/demo/input", response_class=HTMLResponse)
async def search_page(request: Request):
    if frontend_available():
        return render_frontend()
    return render_template(request, "input.html")


@app.get("/about", response_class=HTMLResponse, name="about")
async def about_page():
    if frontend_available():
        return render_frontend()
    return RedirectResponse(url="/", status_code=302)


@app.post("/search")
@app.post("/demo")
@app.post("/demo/input")
async def search_submit(request: Request, text: str = Form(default=""), use_agent: str | None = Form(default=None)):
    normalized_text = (text or "").strip() or DEFAULT_HOME_QUERY
    query = urlencode({"text": normalized_text, "use_agent": "1" if use_agent else "0"})
    redirect_url = f"/search/results?{query}"
    return RedirectResponse(url=redirect_url, status_code=303)


@app.get("/search/results", response_class=HTMLResponse, name="search_outcome")
@app.get("/demo/outcome", response_class=HTMLResponse)
async def outcome(request: Request, text: str, use_agent: str = "0"):
    if frontend_available():
        return render_frontend()
    agent_enabled = use_agent == "1"
    mcp_server_url = INTERNAL_MCP_SERVER_URL or f"{request.base_url}agent-tools/mcp".rstrip("/")
    result = await run_in_threadpool(network.recommend, text, 12, agent_enabled, mcp_server_url)
    return render_template(
        request,
        "outcome.j2",
        text=text,
        query_used=result.get("query_used", text),
        agent_enabled=agent_enabled,
        agent_summary=result.get("agent_summary"),
        agent_error=result.get("agent_error"),
        agent_model=result.get("agent_model"),
        candidates=result["recommendations"],
    )


@app.get("/movie", response_class=HTMLResponse, name="movie_detail")
async def movie_detail(request: Request, title: str, use_agent: str = "0"):
    if frontend_available():
        return render_frontend()
    movie = await run_in_threadpool(network.lookup_movie, title)
    if movie is None:
        return RedirectResponse(url=f"/search/results?{urlencode({'text': title, 'use_agent': use_agent})}", status_code=302)

    related = await run_in_threadpool(network.similar_movies, movie["title"], 6)
    return render_template(
        request,
        "movie.j2",
        movie=movie,
        related=related,
        agent_enabled=(use_agent == "1"),
    )


# --- API Endpoints for React Frontend ---

@app.get("/api/config")
async def get_config():
    status = await run_in_threadpool(network.health_status)
    return {
        "default_query": DEFAULT_HOME_QUERY,
        "retriever": status.get("retriever", {}),
        "agent": status.get("agent", {}),
    }


@app.get("/api/search")
async def api_search(request: Request, text: str, use_agent: bool = False):
    mcp_server_url = INTERNAL_MCP_SERVER_URL or f"{request.base_url}agent-tools/mcp".rstrip("/")
    result = await run_in_threadpool(network.recommend, text, 12, use_agent, mcp_server_url)
    return result


@app.get("/api/movie")
async def api_movie(title: str, use_agent: bool = False):
    movie = await run_in_threadpool(network.lookup_movie, title)
    if movie is None:
        return JSONResponse({"error": "Movie not found"}, status_code=404)

    related = await run_in_threadpool(network.similar_movies, movie["title"], 6)
    return {
        "movie": movie,
        "related": related,
        "agent_enabled": use_agent
    }
