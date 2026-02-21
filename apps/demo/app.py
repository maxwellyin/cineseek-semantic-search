from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

try:
    from . import network
except ImportError:
    import network


APP_DIR = Path(__file__).resolve().parent
app = FastAPI()
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


def render_template(request: Request, template_name: str, **context):
    return templates.TemplateResponse(request, template_name, context)


@app.get("/", response_class=HTMLResponse, name="home")
@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    return render_template(request, "home.html")


@app.get("/demo", response_class=HTMLResponse, name="demo")
@app.get("/demo/input", response_class=HTMLResponse)
async def demo(request: Request):
    return render_template(request, "input.html")


@app.post("/demo")
@app.post("/demo/input")
async def demo_submit(request: Request, text: str = Form(...)):
    query = urlencode({"text": text})
    redirect_url = f"{request.url_for('outcome')}?{query}"
    return RedirectResponse(url=redirect_url, status_code=303)


@app.get("/demo/outcome", response_class=HTMLResponse, name="outcome")
async def outcome(request: Request, text: str):
    result = network.recommend(text)
    return render_template(
        request,
        "outcome.j2",
        text=text,
        candidates=result["recommendations"],
    )
