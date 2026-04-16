class LLMException(Exception):
    """Base exception cho toàn bộ LLM module"""

    def __init__(self, message: str, model: str = None, original_error: Exception = None):
        self.model = model
        self.original_error = original_error
        full_msg = f"[LLM Error] {message}"
        if model:
            full_msg += f" | Model: {model}"
        if original_error:
            full_msg += f" | Original: {type(original_error).__name__} - {original_error}"
        super().__init__(full_msg)