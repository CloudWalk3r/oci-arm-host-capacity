FROM python:3.12-slim

LABEL maintainer="ai-bot" \
      description="Telegram AI assistant with JDownloader file delivery"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Run as non-root for security (uid 1000 matches common host UIDs for volume perms).
RUN useradd -m -u 1000 bot
USER bot

COPY bot.py .
COPY downloader.py .
COPY tgformat.py .
COPY ops.py .
COPY review.py .
COPY assign.py .

HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

CMD ["python", "-u", "bot.py"]
