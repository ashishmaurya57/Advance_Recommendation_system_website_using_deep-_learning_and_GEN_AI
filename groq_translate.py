import polib
import requests
import os

# === CONFIG ===
GROQ_API_URL = "https://api.groq.com/v1/translate"  # Example endpoint, replace with actual
API_KEY = "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"

# Supported languages with target language codes matching your po folders
LANG_CODES = {
    "hi": "Hindi",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
}

LOCALE_DIR = "locale"


def translate_text(text, target_language):
    """
    Sends a text to Groq API to translate to target_language.
    Replace this with actual Groq API call.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "source_language": "en",
        "target_language": target_language,
        "text": text,
    }

    response = requests.post(GROQ_API_URL, json=payload, headers=headers)

    if response.status_code == 200:
        data = response.json()
        return data.get("translation", "")
    else:
        print(f"Error {response.status_code} for text: {text}")
        return ""


def process_po_file(po_filepath, lang_code):
    po = polib.pofile(po_filepath)
    updated = False

    for entry in po:
        # Only translate if msgstr is empty (not translated yet)
        if not entry.msgstr.strip() and entry.msgid.strip():
            translated_text = translate_text(entry.msgid, LANG_CODES[lang_code])
            if translated_text:
                entry.msgstr = translated_text
                updated = True
                print(f"Translated '{entry.msgid}' to '{translated_text}' for {lang_code}")

    if updated:
        po.save()
        print(f"✅ Updated {po_filepath}")
    else:
        print(f"No updates for {po_filepath}")


def main():
    for lang_code in LANG_CODES:
        po_path = os.path.join(LOCALE_DIR, lang_code, "LC_MESSAGES", "django.po")
        if os.path.exists(po_path):
            process_po_file(po_path, lang_code)
        else:
            print(f"PO file not found: {po_path}")


if __name__ == "__main__":
    main()
