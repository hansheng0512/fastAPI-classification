from fastapi import FastAPI, HTTPException, UploadFile, File
import uvicorn
import sys
import logging
from PIL import Image
import io

# local import
from classifier.utils import base64_str_to_PILImage

from classifier.classifier import AlexNet
from classifier.model import Base64str, ResponseDataModelImg, ResponseDataModelBase64

app = FastAPI()
image_classifer = AlexNet()


@app.get("/")
def home():
    return "Hello!"


@app.post("/predict", response_model=ResponseDataModelBase64)
def predict(payload: Base64str):
    try:
        image = base64_str_to_PILImage(payload.base64str)
        predicted_class = image_classifer.predict(image)
        logging.info(f"Predicted Class: {predicted_class}")

        return {
            "likely_class": predicted_class,
        }

    except Exception as error:
        logging.exception(error)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-img", response_model=ResponseDataModelImg)
async def predict(file: UploadFile = File(...)):
    if file.content_type.startswith("image/") is False:
        raise HTTPException(
            status_code=400, detail=f"File '{file.filename}' is not an image."
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        predicted_class = image_classifer.predict(image)

        logging.info(f"Predicted Class: {predicted_class}")
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "likely_class": predicted_class,
        }

    except Exception as error:
        logging.exception(error)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app", reload=True, host="0.0.0.0", port=8000, log_level="info"
    )
