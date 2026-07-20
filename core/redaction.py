import re
from typing import Dict, List


class EntityRedactor:
    """Deterministic PII redaction that works without external NLP services."""

    PATTERNS = {
        "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
        "phone": re.compile(r"\b(?:\+?\d[\d\s().-]{7,}\d)\b"),
        "credit_card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "dosage": re.compile(r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml|iu|units)\b", re.IGNORECASE),
        "date_of_birth": re.compile(r"\b(?:dob|date of birth)[:\s]+[A-Z0-9,/\-\s]{4,25}\b", re.IGNORECASE),
        "account_id": re.compile(r"\b(?:account|acct|member|patient)\s?(?:id|number|no)[:\s#-]+[A-Z0-9-]{4,}\b", re.IGNORECASE),
    }

    def redact(self, text: str) -> Dict[str, object]:
        redacted = text or ""
        findings: List[Dict[str, str]] = []

        for entity_type, pattern in self.PATTERNS.items():
            matches = list(pattern.finditer(redacted))
            for match in matches:
                findings.append({"type": entity_type, "value": match.group(0)})
            redacted = pattern.sub(f"[REDACTED_{entity_type.upper()}]", redacted)

        return {
            "redacted_text": redacted,
            "findings": findings,
            "redaction_count": len(findings),
        }
