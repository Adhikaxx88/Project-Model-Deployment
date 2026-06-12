from fastapi import APIRouter

from api.schemas.predict_schema import PredictRequest, PredictResponse
from api.services.inference_service import InferenceService

router = APIRouter()
inference_service = InferenceService()


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    result = inference_service.predict(request.model_dump())
    return result
