from core.redaction import EntityRedactor


def test_redacts_common_sensitive_entities():
    result = EntityRedactor().redact(
        "Email a@test.com, call 555-123-4567, card 4111 1111 1111 1111, patient id ABCD-1234."
    )

    assert result["redaction_count"] >= 4
    assert "a@test.com" not in result["redacted_text"]
    assert "555-123-4567" not in result["redacted_text"]
    assert "4111 1111 1111 1111" not in result["redacted_text"]
