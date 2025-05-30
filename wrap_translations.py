import os
import re

# 🛠️ Folder where your Django HTML templates are located
TEMPLATE_DIR = 'templates'

TARGET_TAGS = ['p', 'span', 'a', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'button', 'label', 'td', 'th']

def wrap_with_trans(text):
    # If already translated, skip
    if '{% trans' in text or '{% blocktrans' in text:
        return text

    for tag in TARGET_TAGS:
        pattern = fr'<{tag}([^>]*)>([^<>{{}}]+)</{tag}>'
        
        def replacer(match):
            attrs = match.group(1)
            content = match.group(2).strip()
            if not content:
                return match.group(0)
            return f'<{tag}{attrs}>{{% trans "{content}" %}}</{tag}>'

        text = re.sub(pattern, replacer, text)
    
    return text


def process_template_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        original = file.read()

    processed = wrap_with_trans(original)

    if original != processed:
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(processed)
        print(f"✅ Updated: {filepath}")


def walk_templates(dir_path):
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.html'):
                full_path = os.path.join(root, file)
                process_template_file(full_path)


if __name__ == "__main__":
    walk_templates(TEMPLATE_DIR)
    print("✅ Finished auto-wrapping translatable strings.")
