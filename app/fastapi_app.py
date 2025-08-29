"""FastAPI web application for MNIST digit prediction.

Provides a web interface for uploading images and getting predictions
from the trained MNIST model.
"""

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from inference import predict_image

app = FastAPI()

# Initialize template engine
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request) -> HTMLResponse:
    """Render the index page with no prediction.
    
    Args:
        request: The FastAPI request object.
        
    Returns:
        HTMLResponse: The rendered index page.
    """
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

@app.post("/predict/", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)) -> HTMLResponse:
    """Process the uploaded image file, predict its class, and render the result.
    
    Args:
        request: The FastAPI request object.
        file: The uploaded image file.
        
    Returns:
        HTMLResponse: The rendered page with prediction result.
    """
    image_bytes = await file.read()
    prediction = predict_image(image_bytes)
    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host="127.0.0.1", port=8000, reload=True)
