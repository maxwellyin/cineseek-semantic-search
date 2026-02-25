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

## 2. Clone the repository

```bash
git clone https://github.com/maxwellyin/cineseek-semantic-search.git
cd cineseek-semantic-search
```

## 3. Configure environment

Copy the example env file:

```bash
cp .env.example .env
```

Edit `.env` and set:

```bash
GOOGLE_API_KEY=your_gemini_api_key
FLCR_AGENT_PROVIDER=gemini
FLCR_GEMINI_MODEL=gemini-2.5-flash
```

## 4. Build and start the app

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
- Gemini API is used for the agent layer, which is much more practical than running a local LLM on a low-cost CPU VPS.
