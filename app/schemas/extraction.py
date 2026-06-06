from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ExtractedField(BaseModel):
    field_name: str
    value: Optional[str] = None
    confidence: float = 0.0
    evidence_chunk_ids: List[str] = Field(default_factory=list)
    evidence_text: List[str] = Field(default_factory=list)
    extraction_method: str = "deterministic"


class ExtractionResult(BaseModel):
    case_id: str
    event_type: str
    extracted_fields: Dict[str, ExtractedField]
    raw_notes: Dict[str, Any] = Field(default_factory=dict)