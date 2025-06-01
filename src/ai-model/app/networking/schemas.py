from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """
    Response schema for the prediction API.

    This model defines the structure and validation of the response returned by the endpoint. It ensures consistent and
    documented outputfor API clients such as frontend applications or external systems.

    Attributes:
        label (str): The predicted class label from the model.
        confidence (float): The model's confidence in its prediction.
    """

    label: str = Field(
        ...,
        example="benign",
        description="The predicted class label."
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        example=0.87,
        description="Prediction confidence score between 0.0 and 1.0."
    )
