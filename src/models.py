from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

# Input prompt
class PromptItem(BaseModel):
    prompt: str

# Function definitions
class ParameterProperty(BaseModel):
    type: str
    description: Optional[str] = None

class FunctionReturns(BaseModel):
    type: str

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterProperty]
    returns: FunctionReturns

# Output
class FunctionCallOutput(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]
