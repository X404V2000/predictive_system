# Movement prediction service.
#
# The image installs only pinned runtime dependencies and does NOT include the
# motion-capture dataset. Without it the physics endpoints work normally and
# /predict/action reports 503 with an explanation; that keeps the image small and
# the dataset optional. Mount it at /app/training_data to enable action
# recognition (see the README).

FROM python:3.13-slim

# Fail fast and surface errors in build logs rather than mid-run.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# No compiler is installed. Requirements are pure-Python wheels plus numpy, which
# ships manylinux wheels for this platform, so build-essential would add hundreds
# of megabytes for nothing. If a dependency ever needs compiling, the build fails
# here and that is the signal to revisit this decision deliberately.
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# Only application source is copied. .dockerignore excludes the virtualenv, the
# dataset and caches, so this stays small and reproducible.
COPY *.py ./
COPY tools/ ./tools/

# Run as an unprivileged user. The application writes nothing to disk, so it needs
# no ownership of the source tree.
RUN useradd --create-home --shell /usr/sbin/nologin appuser
USER appuser

EXPOSE 8000

# The action model trains during startup, which takes roughly 25 seconds on slow
# hardware. The start period accounts for that so orchestration does not treat a
# still-warming container as unhealthy and restart it in a loop.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/openapi.json', timeout=4).status == 200 else 1)"]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
