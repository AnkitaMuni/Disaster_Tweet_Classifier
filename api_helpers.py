from googletrans import Translator # type: ignore

def translate_text(text, target_language="en"):
    """
    Translates the input text to the target language using Google Translate (via googletrans).
    
    Args:
        text (str): The text to translate.
        target_language (str): The target language code (default is "en" for English).
    
    Returns:
        str: Translated text or the original text if translation fails.
    """
    translator = Translator()
    try:
        translated = translator.translate(text, dest=target_language)
        return translated.text
    except Exception as e:
        print(f"Error during translation: {e}")
        return text