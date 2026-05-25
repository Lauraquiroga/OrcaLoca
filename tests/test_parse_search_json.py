from Orcar.output_parser import load_llm_json

# Test 1: Valid JSON
def test_load_llm_json_valid():
    input_text = '{"a": 1, "b": "test"}'
    result = load_llm_json(input_text)
    assert result == {"a": 1, "b": "test"}

# Test 2: JSON with problematic escapes (should be fixed by fix_escapes)
def test_load_llm_json_escape_fix():
    # This string simulates a common LLM output with problematic escapes
    input_text = r'{"pattern": "READ [TS]ERR(\s+[0-9]+)+"}'
    result = load_llm_json(input_text)
    assert result["pattern"].startswith("READ [TS]ERR(")
    assert "+)+" in result["pattern"]

# Test 3: Malformed JSON (should be repaired)
def test_load_llm_json_repair():
    # Missing closing brace, should be repaired
    input_text = '{"a": 1, "b": 2'
    result = load_llm_json(input_text)
    assert result["a"] == 1
    assert result["b"] == 2

# Test 4: Completely invalid JSON (should raise ValueError)
def test_load_llm_json_failure():
    input_text = 'not a json at all'
    try:
        load_llm_json(input_text)
        print("test_load_llm_json_failure failed: did not raise ValueError")
    except ValueError:
        print("test_load_llm_json_failure passed")

# Test 5: JSON with mixed quotes (should be repaired)
def test_load_llm_json_mixed_quotes():
    input_text = '{"a":\'e\'}'
    result = load_llm_json(input_text)
    assert result["a"] == "e"

if __name__ == "__main__":
    test_load_llm_json_valid()
    test_load_llm_json_escape_fix()
    test_load_llm_json_repair()
    test_load_llm_json_failure()
    test_load_llm_json_mixed_quotes()