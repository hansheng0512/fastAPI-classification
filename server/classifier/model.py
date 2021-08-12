from pydantic import BaseModel

class ResponseDataModelImg(BaseModel):
    filename: str
    content_type: str
    likely_class: str


class ResponseDataModelBase64(BaseModel):
    likely_class: str


class Base64str(BaseModel):
    base64str: str