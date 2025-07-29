import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add project root to path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from src.pdf_processor import extract_text_from_pdf, chunk_text
from src.rag_pipeline import initialize_rag, query_rag
from src.models import QueryRequest
import shutil

app = FastAPI()
templates = Jinja2Templates(directory="src/templates")
app.mount("/static", StaticFiles(directory="src/templates"), name="static")

# Directory to store uploaded PDFs
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Extract and process PDF
    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)
    initialize_rag(chunks)
    
    return {"message": f"PDF {file.filename} uploaded and processed successfully"}

@app.post("/query")
async def query_pdf(request: QueryRequest):
    answer = query_rag(request.query)
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    port = 8000
    url = f"http://127.0.0.1:{port}/"  # Changed from /docs to /
    webbrowser.open(url)
    uvicorn.run(app, host="0.0.0.0", port=port)