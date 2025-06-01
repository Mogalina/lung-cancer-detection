import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.networking.schemas import PredictionResponse
from app.utils.utils import read_image
from typing import Any
from app.services.model_service import LungCancerModelService

logger: logging.Logger = logging.getLogger(__name__)

# Initialize FastAPI app
app: FastAPI = FastAPI(title="Lung Cancer Detection API", version="1.0")

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate the model service with the given configs file.
model_service: LungCancerModelService = LungCancerModelService(config_path="config.yaml")
model_service.load_model()

@app.get("/")
def root() -> dict[str, str]:
    """
    Root endpoint to check if the API is running.

    Returns:
        dict[str, str]: A JSON response confirming the API status.
    """
    return {"message": "Lung Cancer Detection API is running"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """
    Accepts an image file and returns a lung cancer prediction.

    Args:
        file (UploadFile): The image file uploaded by the user. Supported formats: jpg, jpeg, png.

    Returns:
        PredictionResponse: A response model containing prediction results.

    Raises:
        HTTPException: If file format is invalid or prediction fails.
    """
    # Validate file extension
    filename: str = file.filename.lower()
    if not filename.endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Supported types: jpg, jpeg, png."
        )

    try:
        # Read the file contents asynchronously
        contents: bytes = await file.read()

        # Convert raw bytes to a tensor or image ready for prediction
        img: Any = read_image(contents)

        # Use model service to get the prediction result
        result: dict[str, Any] = model_service.predict(img)

        # Return response as defined in the schema
        return PredictionResponse(**result)

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
