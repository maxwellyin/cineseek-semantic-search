# Deploying CineSeek on Vultr

This guide assumes a small Ubuntu VPS with Docker installed.

## 1. Install Docker

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
```

Verify:

```bash
docker --version
docker compose version
```

## 2. Fastest path: pull the prebuilt image

```bash
docker pull ghcr.io/maxwellyin/cineseek-semantic-search:latest
```

Run it:

```bash
docker run -d \
  --name cineseek \
  -p 8000:8000 \
  -e GROQ_API_KEY=your_groq_api_key \
  ghcr.io/maxwellyin/cineseek-semantic-search:latest
```

## 3. Alternative: deploy from source

If you prefer to build on the server:

```bash
git clone https://github.com/maxwellyin/cineseek-semantic-search.git
cd cineseek-semantic-search
cp .env.example .env
```

Edit `.env` and set:

```bash
GROQ_API_KEY=your_groq_api_key
FLCR_AGENT_PROVIDER=groq
FLCR_GROQ_MODEL=qwen/qwen3-32b
```

Or use Gemini instead:

```bash
GOOGLE_API_KEY=your_gemini_api_key
FLCR_AGENT_PROVIDER=gemini
FLCR_GEMINI_MODEL=gemini-2.5-flash-lite
```

## 4. Build and start the app from source

```bash
docker compose up -d --build
```

Check logs:

```bash
docker compose logs -f
```

## 5. Open the app

By default the app listens on port `8000`.

Open:

```text
http://YOUR_SERVER_IP:8000/search
```

## 6. Recommended firewall rules

Allow:

- `22/tcp` for SSH
- `8000/tcp` for the app

If you later put Nginx in front, expose only:

- `80/tcp`
- `443/tcp`

## 7. Updating after new commits

```bash
git pull
docker compose up -d --build
```

## Notes

- This image already contains the processed dataset, model cache, trained checkpoint, and FAISS index.
- The server does **not** need to run training or preprocessing.
- Groq is the default hosted agent backend for this deployment.
- Gemini is also supported if you prefer it over Groq.
