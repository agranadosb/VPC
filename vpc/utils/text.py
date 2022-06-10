def text_ellipsis(text: str, max_size: int = 8) -> str:
    """If a text is longer than `max_size`, it will be truncated and ellipsis
    will be added.

    Parameters
    ----------
    text: str
        Text where the ellipsis would be added if needed.
    max_size: int = 8
        Maximum size of the text.

    Returns
    -------
    The with ellipsis added if needed."""
    if text:
        ellipsis_text = ""
        if len(text) > max_size:
            ellipsis_text = "..."
        text = f"{text[:max_size]}{ellipsis_text}"
    return text
