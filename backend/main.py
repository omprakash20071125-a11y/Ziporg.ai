import os
import uuid
import tempfile
import shutil
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from phase3 import run_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)


@app.post("/generate")
def generate(
    prompt: str = Form(...),
    reference_image: Optional[UploadFile] = File(None),
):
    reference_image_path = ""
    image_tmp_dir = None

    # Save the uploaded reference image to disk, since run_pipeline expects a path
    if reference_image is not None:
        image_tmp_dir = tempfile.mkdtemp()
        ext = os.path.splitext(reference_image.filename or "")[1] or ".png"
        reference_image_path = os.path.join(image_tmp_dir, f"reference{ext}")
        try:
            with open(reference_image_path, "wb") as f:
                shutil.copyfileobj(reference_image.file, f)
        except Exception as e:
            shutil.rmtree(image_tmp_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail=f"Failed to save reference image: {e}")

    try:
        file_code = run_pipeline(prompt, reference_image_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")
    finally:
        if image_tmp_dir:
            shutil.rmtree(image_tmp_dir, ignore_errors=True)

    if not file_code:
        raise HTTPException(status_code=500, detail="No files were generated")

    work_dir = tempfile.mkdtemp()
    try:
        for filename, content in file_code.items():
            fpath = os.path.join(work_dir, filename)
            os.makedirs(os.path.dirname(fpath) or work_dir, exist_ok=True)
            with open(fpath, "w") as f:
                f.write(content)

        run_id = str(uuid.uuid4())
        zip_base = os.path.join(tempfile.gettempdir(), run_id)
        zip_path = shutil.make_archive(zip_base, "zip", work_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Zip creation failed: {e}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    return FileResponse(
        zip_path,
        filename="project.zip",
        media_type="application/zip",
    )
