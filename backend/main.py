import os
import uuid
import tempfile
import shutil
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from phase1 import run_pipeline
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ziporg-ai.vercel.app"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptIn(BaseModel):
    prompt: str


@app.post("/generate")
def generate(body: PromptIn):
    try:
        file_code = run_pipeline(body.prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")

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
