def apply_stop_strings(text: str, stop: list[str]) -> str:
    stop_positions = [idx for stop_text in stop if (idx := text.find(stop_text)) != -1]
    if not stop_positions:
        return text
    return text[: min(stop_positions)]
