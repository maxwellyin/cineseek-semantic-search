# Deploying CineSeek on a VPS

This guide assumes a small Ubuntu VPS with Docker installed. The public deployment is served behind a custom HTTPS domain:

```text
https://cineseek.maxwellyin.com
```

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

By default the container listens on port `8000`.

For local server verification:

```text
http://YOUR_SERVER_IP:8000/search
```

For the public deployment, put a reverse proxy such as Nginx or Caddy in front of the container and open:

```text
https://cineseek.maxwellyin.com/search
```

## 6. Recommended firewall rules

During direct IP testing, allow:

- `22/tcp` for SSH
- `8000/tcp` for the app

For the public domain deployment, expose only:

- `80/tcp`
- `443/tcp`

## 7. Updating after new commits

```bash
git pull
docker compose up -d --build
```

## 8. Optional GitHub Actions CI/CD

This repository can also deploy automatically after every push to `main`.

Add these GitHub Actions secrets:

- `VPS_HOST`
- `VPS_USERNAME`
- `VPS_SSH_KEY`
- `VPS_PORT` (optional, defaults to `22`)

The pipeline then does:

1. run lightweight checks
2. build and push a `linux/amd64` image to GHCR
3. SSH into the VPS
4. run the existing server-side deploy script (`/root/cineseek_deploy.sh`)

This assumes your server already has a working `.env` file that the deploy script uses for runtime secrets such as `GROQ_API_KEY`.

Old GHCR image versions are pruned separately by the `Prune GHCR Images` workflow.

## Notes

- This image already contains the processed dataset, local sentence-transformer cache, and FAISS index.
- The server does **not** need to run training or preprocessing.
- Groq is the default hosted agent backend for this deployment.
- Gemini is also supported if you prefer it over Groq.
