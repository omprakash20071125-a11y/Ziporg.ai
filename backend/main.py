import os
import base64
import uuid
import logging
import tempfile
import shutil
from typing import Optional
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Import your LangGraph setup and file generation pipeline
from chat_pipeline_new import build_graph
from phase4 import run_pipeline

# ---------------------------------------------------------------------
# Logging: full detail goes here (server-side only), never to the client
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ziporg")

app = FastAPI(title="Unified Assistant & Code Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        # Actual deployed frontend
        "https://ziporg-ai.vercel.app",
    ],
    # Also match any preview deployments for this project,
    # e.g. https://ziporg-ai-<hash>-<team>.vercel.app
    allow_origin_regex=r"https://ziporg-ai.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

# Initialize LangGraph once at app startup. Its InMemorySaver checkpointer
# keeps each thread_id's parsed/chunked project + chat_history alive in
# memory across requests.
graph_app = build_graph()

# ---------------------------------------------------------------------
# Limits (tune these as you like — these are sane free-tier defaults)
# ---------------------------------------------------------------------
MAX_PROMPT_LEN = 5000             # characters
MAX_QUERY_LEN = 2000               # characters
MAX_IMAGE_SIZE = 5 * 1024 * 1024   # 5 MB
ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

GENERIC_ERROR = "Can't generate a response at this moment due to some technical issue. Please try again shortly."

# ---------------------------------------------------------------------
# In-memory job store for /generate background jobs.
#
# jobs[job_id] = {
#     "status": "running" | "done" | "error",
#     "zip_path": str | None,
#     "detail": str | None,
# }
#
# NOTE: This is process-local memory. If Railway restarts your instance
# or runs multiple workers/replicas, jobs won't be visible across them.
# For a single-instance deployment (the common case on Railway's free/
# hobby tiers) this is fine. If you scale to multiple workers later,
# swap this dict for Redis or a database table.
# ---------------------------------------------------------------------
jobs: dict[str, dict] = {}


# ---------------------------------------------------------------------
# Helper: detect quota / rate-limit style errors across providers
# ---------------------------------------------------------------------
def is_rate_limit_error(e: Exception) -> bool:
    """Best-effort detection of quota/rate-limit errors from Gemini, Cohere,
    or any generic HTTP 429, without depending on a specific SDK's
    exception classes (keeps this resilient to library changes)."""
    msg = str(e).lower()
    return any(
        term in msg
        for term in [
            "429",
            "quota",
            "rate limit",
            "rate_limit",
            "resource_exhausted",
            "resource exhausted",
            "too many requests",
        ]
    )


# ---------------------------------------------------------------------
# Helper: safe path join to prevent path traversal when writing
# generated files (protects against a malicious/hallucinated filename
# like "../../etc/x" or an absolute path escaping work_dir)
# ---------------------------------------------------------------------
def safe_join(base: str, filename: str) -> str:
    filename = filename.replace("\\", "/").lstrip("/")
    target = os.path.normpath(os.path.join(base, filename))
    base_abs = os.path.abspath(base)
    if not (target == base_abs or target.startswith(base_abs + os.sep)):
        raise ValueError(f"Unsafe file path rejected: {filename}")
    return target


# ---------------------------------------------------------------------
# Global safety net: catches anything that slips past individual
# try/excepts below, so raw exception text is NEVER returned to a client.
# ---------------------------------------------------------------------
@app.exception_handler(Exception)
async def catch_all_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": GENERIC_ERROR})


# --- Pydantic Schema for Q&A About a Generated Project ---
class ChatPayload(BaseModel):
    query: str
    thread_id: str
    # Only required on the FIRST /chat call for a given thread_id — after
    # that, check_for_chunks in chat_pipeline.py sees the cached chunks for
    # this thread and skips straight to retrieval, so it can be omitted
    # (the frontend sends it as null on later calls).
    project_zip_base64: Optional[str] = None


# =====================================================================
# 1. THE CHAT ENDPOINT — Q&A about a project AFTER it's been generated
# =====================================================================
@app.post("/chat")
async def project_chat(payload: ChatPayload):
    """
    Q&A about a generated project. thread_id must match the thread_id used
    when that project was generated (or any thread_id you've since chatted
    with). On the first message for a thread_id, project_zip_base64 must be
    supplied so the graph has something to unzip and index; later messages
    reuse the same thread_id's cached chunks + chat_history automatically.
    """
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if len(payload.query) > MAX_QUERY_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"Query too long (max {MAX_QUERY_LEN} characters).",
        )

    inputs = {"query": payload.query}
    if payload.project_zip_base64:
        inputs["zip_path"] = payload.project_zip_base64

    config = {"configurable": {"thread_id": payload.thread_id}}

    try:
        result = graph_app.invoke(inputs, config=config)

    except KeyError:
        raise HTTPException(
            status_code=400,
            detail="No project is loaded for this thread yet — attach project_zip_base64 on the first message.",
        )

    except (base64.binascii.Error, ValueError) as e:
        logger.warning(f"Bad base64/zip for thread={payload.thread_id}: {e}")
        raise HTTPException(
            status_code=400,
            detail="Project data could not be read — please try generating again.",
        )

    except Exception as e:
        # Full detail logged server-side only; client gets a safe, generic message.
        logger.error(f"Chat error [thread={payload.thread_id}]: {e}", exc_info=True)
        if is_rate_limit_error(e):
            raise HTTPException(status_code=503, detail=GENERIC_ERROR)
        raise HTTPException(status_code=500, detail=GENERIC_ERROR)

    return {
        "thread_id": payload.thread_id,
        "response": result.get("response", "I couldn't generate an answer for that."),
    }


# =====================================================================
# 2. GENERATION — background job + polling
# =====================================================================

def run_generation_job(
    job_id: str,
    prompt: str,
    reference_image_path: str,
    image_tmp_dir: Optional[str],
):
    """Runs the (slow) pipeline in a background thread and stores the
    result/error in `jobs`. This is what lets /generate return instantly
    instead of holding the HTTP connection open for the whole pipeline."""
    try:
        file_code = run_pipeline(prompt, reference_image_path)
    except Exception as e:
        logger.error(f"Pipeline failed [job={job_id}]: {e}", exc_info=True)
        detail = GENERIC_ERROR
        if is_rate_limit_error(e):
            detail = GENERIC_ERROR  # keep message generic either way to the client
        jobs[job_id] = {"status": "error", "zip_path": None, "detail": detail}
        return
    finally:
        if image_tmp_dir:
            shutil.rmtree(image_tmp_dir, ignore_errors=True)

    if not file_code:
        jobs[job_id] = {
            "status": "error",
            "zip_path": None,
            "detail": "No files were generated. Please try again.",
        }
        return

    work_dir = tempfile.mkdtemp()
    try:
        for filename, content in file_code.items():
            try:
                fpath = safe_join(work_dir, filename)
            except ValueError:
                logger.warning(f"Rejected unsafe filename from pipeline [job={job_id}]: {filename}")
                continue  # skip this file rather than failing the whole generation

            os.makedirs(os.path.dirname(fpath) or work_dir, exist_ok=True)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(str(content))

        zip_base = os.path.join(tempfile.gettempdir(), job_id)
        zip_path = shutil.make_archive(zip_base, "zip", work_dir)

        jobs[job_id] = {"status": "done", "zip_path": zip_path, "detail": None}

    except Exception as e:
        logger.error(f"Zip creation failed [job={job_id}]: {e}", exc_info=True)
        jobs[job_id] = {"status": "error", "zip_path": None, "detail": GENERIC_ERROR}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/generate")
def generate(
    background_tasks: BackgroundTasks,
    thread_id: str = Form(...),  # frontend generates this and reuses it for later /chat calls
    prompt: str = Form(...),
    reference_image: Optional[UploadFile] = File(None),
):
    """Validates input, saves any reference image, enqueues the pipeline as
    a background task, and returns a job_id IMMEDIATELY — it does not wait
    for the pipeline to finish. The frontend polls /generate/status/{job_id}
    and then downloads from /generate/download/{job_id} once done."""

    # -----------------------------
    # Validate prompt
    # -----------------------------
    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    if len(prompt) > MAX_PROMPT_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt too long (max {MAX_PROMPT_LEN} characters).",
        )

    reference_image_path = ""
    image_tmp_dir = None

    # -----------------------------
    # Validate + save uploaded image
    # -----------------------------
    if reference_image is not None:
        ext = os.path.splitext(reference_image.filename or "")[1].lower() or ".png"
        if ext not in ALLOWED_IMAGE_EXTS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported image type. Allowed: {', '.join(ALLOWED_IMAGE_EXTS)}",
            )

        contents = reference_image.file.read()
        if len(contents) > MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Image too large (max {MAX_IMAGE_SIZE // (1024*1024)}MB).",
            )
        reference_image.file.seek(0)

        image_tmp_dir = tempfile.mkdtemp()
        reference_image_path = os.path.join(image_tmp_dir, f"reference{ext}")
        try:
            with open(reference_image_path, "wb") as f:
                f.write(contents)
        except Exception as e:
            shutil.rmtree(image_tmp_dir, ignore_errors=True)
            logger.error(f"Failed to save reference image [thread={thread_id}]: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=GENERIC_ERROR)

    # -----------------------------
    # Enqueue the pipeline as a background job and return right away
    # -----------------------------
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "running", "zip_path": None, "detail": None}

    background_tasks.add_task(
        run_generation_job, job_id, prompt, reference_image_path, image_tmp_dir
    )

    return {"job_id": job_id, "status": "running"}


@app.get("/generate/status/{job_id}")
def generate_status(job_id: str):
    """Lightweight poll endpoint — the frontend calls this every few
    seconds. Always returns fast regardless of how long the underlying
    pipeline takes."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job_id.")
    return {"status": job["status"], "detail": job["detail"]}


@app.get("/generate/download/{job_id}")
def generate_download(job_id: str, thread_id: str = "project"):
    """Called once /generate/status/{job_id} reports status == 'done'."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job_id.")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job is not finished yet (status: {job['status']}).")
    if not job["zip_path"] or not os.path.exists(job["zip_path"]):
        raise HTTPException(status_code=410, detail="Result is no longer available — please generate again.")

    return FileResponse(
        job["zip_path"],
        filename=f"project_{thread_id}.zip",
        media_type="application/zip",
    )
    if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
