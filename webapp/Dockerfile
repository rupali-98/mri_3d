# Webapp: the cloud upload page (streamlit_dashboard/upload_page.py). A visitor uploads one scan; the page
# sends it to the API (backend/, model from the MLflow bucket) and draws the result. No PyTorch, no data/ or
# runs/: separate from frontend/, the local research viewer, which is unchanged.
# Build in the cloud with deploy/cloudbuild-webapp.yaml, or locally from the project root:
#     docker build -f webapp/Dockerfile -t hippo-webapp .
#     docker run -p 8080:8080 -e HIPPO_API_URL=https://... -e HIPPO_API_KEY=... hippo-webapp
# Research use only, not for diagnosis.
FROM python:3.11.17-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-webapp.txt .
RUN pip install -r requirements-webapp.txt

# Only the modules the page imports (the others need PyTorch)
COPY src/hippo/__init__.py src/hippo/api_client.py src/hippo/examples.py src/hippo/slices.py src/hippo/
COPY streamlit_dashboard/upload_page.py streamlit_dashboard/

RUN useradd --create-home --uid 1000 webapp
USER webapp

ENV PYTHONPATH=/app/src \
    PORT=8080
EXPOSE 8080

# Cloud Run sets PORT; 30 MB matches the API's upload limit
CMD exec streamlit run streamlit_dashboard/upload_page.py --server.address=0.0.0.0 --server.port=${PORT} \
    --server.headless=true --browser.gatherUsageStats=false --server.maxUploadSize=30
