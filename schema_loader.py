def load_schema_from_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()
