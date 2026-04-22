from __future__ import annotations

from pathlib import Path
import re
from urllib.parse import urlencode

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape

try:
    from . import network
except ImportError:
    import network


APP_DIR = Path(__file__).resolve().parent
app = FastAPI()
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")


def render_inline_markdown(value: str | None) -> Markup:
    escaped = escape(value or "")
    html = str(escaped)
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", html)
    html = re.sub(r"`(.+?)`", r"<code>\1</code>", html)
    return Markup(html)


templates.env.filters["inline_markdown"] = render_inline_markdown


def render_template(request: Request, template_name: str, **context):
    return templates.TemplateResponse(request, template_name, context)


@app.get("/", response_class=HTMLResponse, name="home")
@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    return render_template(request, "home.html")


@app.head("/")
async def head_home():
    return Response(status_code=200)


@app.get("/favicon.ico")
@app.get("/apple-touch-icon.png")
@app.get("/apple-touch-icon-precomposed.png")
async def favicon_alias():
    return RedirectResponse(url="/static/favicon.svg", status_code=308)


@app.get("/health")
async def health():
    status = network.health_status()
    status_code = 200 if status.get("status") == "ok" else 503
    return JSONResponse(status, status_code=status_code)


@app.get("/search", response_class=HTMLResponse, name="search")
@app.get("/demo", response_class=HTMLResponse)
@app.get("/demo/input", response_class=HTMLResponse)
async def search_page(request: Request):
    return render_template(request, "input.html")


@app.post("/search")
@app.post("/demo")
@app.post("/demo/input")
async def search_submit(request: Request, text: str = Form(...), use_agent: str | None = Form(default=None)):
    query = urlencode({"text": text, "use_agent": "1" if use_agent else "0"})
    redirect_url = f"/search/results?{query}"
    return RedirectResponse(url=redirect_url, status_code=303)


@app.get("/search/results", response_class=HTMLResponse, name="search_outcome")
@app.get("/demo/outcome", response_class=HTMLResponse)
async def outcome(request: Request, text: str, use_agent: str = "0"):
    agent_enabled = use_agent == "1"
    result = network.recommend(text, use_agent=agent_enabled)
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
