import os
import uuid
import tempfile
import shutil
from typing import Optional

from pydantic import BaseModel
from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Form,
    BackgroundTasks,
)
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from phase3 import run_pipeline
from chat_pipeline import build_graph

app = FastAPI()


@app.get("/")
def root():
    return {"status": "running"}


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


# --- Pydantic Schema for Q&A About a Generated Project ---
class ChatPayload(BaseModel):
    query: str
    thread_id: str
    # Only required on the FIRST /chat call for a given thread_id — after
    # that, check_for_chunks in chat_pipeline.py sees the cached chunks for
    # this thread and skips straight to retrieval, so it can be omitted.
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

    inputs = {"query": payload.query}
    if payload.project_zip_base64:
        inputs["zip_path"] = payload.project_zip_base64

    config = {"configurable": {"thread_id": payload.thread_id}}

    try:
        result = graph_app.invoke(inputs, config=config)
    except KeyError:
        # No cached chunks for this thread_id AND no zip was attached.
        raise HTTPException(
            status_code=400,
            detail="No project is loaded for this thread yet — attach project_zip_base64 on the first message.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {e}")

    return {
        "thread_id": payload.thread_id,
        "response": result.get("response", "I couldn't generate an answer for that.")
    }


# =====================================================================
# 2. THE GENERATION ENDPOINT (Triggers the actual codebase build)
# =====================================================================
@app.post("/generate")
def generate(
    background_tasks: BackgroundTasks,
    thread_id: str = Form(...),  # frontend generates this and reuses it for later /chat calls
    prompt: str = Form(...),
    reference_image: Optional[UploadFile] = File(None),
):

    reference_image_path = ""
    image_tmp_dir = None

    # -----------------------------
    # Save uploaded image
    # -----------------------------

    if reference_image is not None:
        image_tmp_dir = tempfile.mkdtemp()

        ext = os.path.splitext(reference_image.filename or "")[1] or ".png"

        reference_image_path = os.path.join(
            image_tmp_dir,
            f"reference{ext}"
        )

        try:
            with open(reference_image_path, "wb") as f:
                shutil.copyfileobj(reference_image.file, f)

        except Exception as e:
            shutil.rmtree(image_tmp_dir, ignore_errors=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save image: {e}",
            )

    # -----------------------------
    # Run AI pipeline
    # -----------------------------

    try:
        print("Starting pipeline...")

        file_code = run_pipeline(prompt, reference_image_path)

        print(f"Pipeline generated {len(file_code)} files.")

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {e}",
        )

    finally:
        if image_tmp_dir:
            shutil.rmtree(image_tmp_dir, ignore_errors=True)

    if not file_code:
        raise HTTPException(
            status_code=500,
            detail="No files generated.",
        )

    # -----------------------------
    # Write files
    # -----------------------------

    work_dir = tempfile.mkdtemp()

    try:

        for filename, content in file_code.items():

            fpath = os.path.join(work_dir, filename)

            os.makedirs(
                os.path.dirname(fpath),
                exist_ok=True,
            )

            with open(
                fpath,
                "w",
                encoding="utf-8",
            ) as f:

                f.write(str(content))

        print("Files written.")

        # -----------------------------
        # Create ZIP
        # -----------------------------

        run_id = str(uuid.uuid4())

        zip_base = os.path.join(
            tempfile.gettempdir(),
            run_id,
        )

        zip_path = shutil.make_archive(
            zip_base,
            "zip",
            work_dir,
        )

        print("ZIP created:", zip_path)

        size = os.path.getsize(zip_path)

        print(f"ZIP size: {size/1024/1024:.2f} MB")

    except Exception as e:

        shutil.rmtree(work_dir, ignore_errors=True)

        raise HTTPException(
            status_code=500,
            detail=f"Zip creation failed: {e}",
        )

    if not os.path.exists(zip_path):
        raise HTTPException(
            status_code=500,
            detail="ZIP file missing.",
        )

    # -----------------------------
    # Cleanup AFTER response
    # -----------------------------

    background_tasks.add_task(
        shutil.rmtree,
        work_dir,
        ignore_errors=True,
    )

    background_tasks.add_task(
        os.remove,
        zip_path,
    )

    print("Returning ZIP...")

    return FileResponse(
        path=zip_path,
        filename=f"project_{thread_id}.zip",
        media_type="application/zip",
    )
