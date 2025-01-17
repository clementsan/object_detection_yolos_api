from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from .detection import ml_detection, utils
from contextlib import asynccontextmanager
from typing import Optional


def detection(yolos_processor, yolos_model, image_bytes):
    # Object detection
    results = ml_detection.object_detection(yolos_processor, yolos_model, image_bytes)

    # Convert dictionary of tensors to JSON
    result_json = utils.convert_tensor_dict_to_json(results)

    return result_json


# Example with global variable as dict
# ml_yolos = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    # ml_yolos = ml_detection.load_model()
    app.processor_tiny, app.model_tiny = ml_detection.load_model("hustvl/yolos-tiny")
    app.processor_small, app.model_small = ml_detection.load_model("hustvl/yolos-small")

    yield
    # Clean up the ML model and release the resources
    del app.processor_tiny, app.model_tiny
    del app.processor_small, app.model_small
    # ml_yolos.clear()


app = FastAPI(
    lifespan=lifespan,
    title="Object detection",
    description="Object detection on COCO dataset",
    version="1.0",
)


@app.get("/")
def home():
    return {"message": "Welcome to the object detection API"}

@app.get("/status")
def status():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {"name": "object-detection", "description": "object detection on COCO dataset"}


# Detection with optional model type
@app.post("/api/v1/detect")
async def detect(image: UploadFile = File(...), model: Optional[str] = Query(None)):
    # Read the image file
    image_bytes = await image.read()

    print("API ML model: ", model)

    # ML detection
    if (model is None) or (model == "yolos-tiny"):
        output_json = detection(app.processor_tiny, app.model_tiny, image_bytes)
    elif model == "yolos-small":
        output_json = detection(app.processor_small, app.model_small, image_bytes)
    else:
        raise HTTPException(status_code=400, detail="Incorrect model type")
    return output_json
