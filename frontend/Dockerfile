# Frontend: the Streamlit viewer (streamlit_dashboard/compare_slices.py). It predicts in-process with the
# runs in runs/ (it compares several runs with the ground truth, which the API's single model can't), so it
# needs CPU torch too. data/ and runs/ are not copied in; compose.yaml mounts them, so scans stay on the host.
#     docker compose up --build frontend       # then open http://localhost:8501
# Research use only, not for diagnosis.
FROM python:3.11.17-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Same CPU torch layer as backend/Dockerfile, so the build cache can share it
COPY requirements-dashboard.txt .
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements-dashboard.txt

COPY src/ src/
COPY configs/ configs/
COPY streamlit_dashboard/ streamlit_dashboard/

RUN useradd --create-home --uid 1000 viewer
USER viewer

ENV PYTHONPATH=/app/src \
    HIPPO_MLFLOW=0
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_dashboard/compare_slices.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", \
     "--server.runOnSave=true", "--browser.gatherUsageStats=false"]
