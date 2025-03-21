from typing import Any, Dict, List, Optional

from pydantic import Extra, Field

from .base import BaseModel


class MetadataServerResponse(BaseModel):
    name: str
    version: str
    extensions: List[str]


class MetadataServerErrorResponse(BaseModel):
    error: str


class MetadataModelErrorResponse(BaseModel):
    error: str


class Parameters(BaseModel):
    class Config:
        extra = Extra.allow

    content_type: Optional[str] = None
    headers: Optional[Dict[str, Any]] = None


class TensorData(BaseModel):
    __root__: Any = Field(..., title="TensorData")

    def __iter__(self):
        return iter(self.__root__)

    def __getitem__(self, idx):
        return self.__root__[idx]

    def __len__(self):
        return len(self.__root__)


class RequestOutput(BaseModel):
    name: str
    parameters: Optional["Parameters"] = None


class ResponseOutput(BaseModel):
    name: str
    shape: List[int]
    datatype: str
    parameters: Optional["Parameters"] = None
    data: "TensorData"


class InferenceResponse(BaseModel):
    model_name: str
    model_version: Optional[str] = None
    id: Optional[str] = None
    parameters: Optional["Parameters"] = None
    outputs: List["ResponseOutput"]


class InferenceErrorResponse(BaseModel):
    error: Optional[str] = None


class MetadataTensor(BaseModel):
    name: str
    datatype: str
    shape: List[int]
    parameters: Optional["Parameters"] = None


class RequestInput(BaseModel):
    name: str
    shape: List[int]
    datatype: str
    parameters: Optional["Parameters"] = None
    data: "TensorData"


class MetadataModelResponse(BaseModel):
    name: str
    versions: Optional[List[str]] = None
    platform: str
    inputs: Optional[List["MetadataTensor"]] = None
    outputs: Optional[List["MetadataTensor"]] = None
    parameters: Optional["Parameters"] = None


class InferenceRequest(BaseModel):
    id: Optional[str] = None
    parameters: Optional["Parameters"] = None
    inputs: List["RequestInput"]
    outputs: Optional[List["RequestOutput"]] = None