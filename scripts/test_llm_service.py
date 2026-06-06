import sys
from pathlib import Path
from pprint import pprint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from app.services.llm_service import LLMService


def main():
    service = LLMService()

    print("\nTesting Ollama LLM service...")
    print(f"Model: {service.model}")
    print(f"Base URL: {service.base_url}")

    result = service.health_check()

    print("\nHealth check result:")
    pprint(result)

    prompt = """
Return only valid JSON.

Extract the debt financing field from this evidence.

Field name: principal_amount

Evidence:
The Company entered into a credit agreement providing for an aggregate principal amount of $750,000,000.

Return JSON with:
{
  "field_name": "...",
  "value": "...",
  "confidence": 0.0,
  "evidence_quote": "...",
  "reason": "..."
}
"""

    extraction_result = service.generate_json(prompt)

    print("\nExtraction test result:")
    pprint(extraction_result)


if __name__ == "__main__":
    main()