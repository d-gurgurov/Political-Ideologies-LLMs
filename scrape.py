import os
import json
import time
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Mapping language codes to output filenames
LANGUAGES = {
    'bg': 'Български',
    'cz': 'Čeština',
    'de': 'Deutsch',
    'en': 'English',
    'es': 'Español',
    'fr': 'Français',
    'it': 'Italiano',
    'fa': 'فارسی',
    'pl': 'Polski',
    'pt-pt': 'Português',
    'ro': 'Română',
    'ru': 'Русский',
    'sl': 'Slovenščina',
    'tr': 'Türkçe'
}

BASE_URL = "https://www.politicalcompass.org/test/{}"
OUTPUT_DIR = "political_compass_questions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36'
}

def get_page(session, lang_code, page_num, data):
    url = f"https://www.politicalcompass.org/test/{lang_code}"
    response = session.post(url, data=data, headers=HEADERS)

    if response.status_code != 200 or "Page not found" in response.text:
        raise Exception(f"Failed to load page {page_num} for language {lang_code}. Status code: {response.status_code}")

    soup = BeautifulSoup(response.text, 'html.parser')

    questions = []
    choices = []

    # Flexible match for all fieldsets
    fieldsets = soup.find_all("fieldset", class_="b1 pa2 mb1")
    for fieldset in fieldsets:
        # Find the legend text
        legend = fieldset.find("legend")
        if legend:
            questions.append(legend.get_text(strip=True))

        # Only extract choices on page 1 and from the first fieldset
        if page_num == 1 and not choices:
            for label in fieldset.find_all("label"):
                span = label.find("span")
                if span:
                    # Try to extract text after <input>, ignoring whitespace
                    # Works even in RTL (Farsi) because get_text() handles it correctly
                    input_tag = span.find("input")
                    if input_tag and input_tag.next_sibling:
                        text = input_tag.next_sibling.strip()
                        choices.append(text)

    form = soup.find("form")
    if form is None:
        error_path = f"error_{lang_code}_page{page_num}.html"
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        raise Exception(f"Form not found on page {page_num} for {lang_code}. Saved to {error_path}")

    # Create new POST data for the next page
    next_data = {input_.get("name"): input_.get("value", "") for input_ in form.find_all("input") if input_.get("name")}
    next_data["page"] = str(page_num + 1)

    for key in next_data:
        if key and key not in ("page", "carried_ec", "carried_soc", "populated"):
            next_data[key] = "1"  # default choice

    return questions, choices if page_num == 1 else None, next_data

def scrape_language(lang_code):
    session = requests.Session()
    data = {
        "page": "1"
    }

    all_questions = {}
    all_choices = None

    for i in range(1, 7):
        try:
            questions, choices, data = get_page(session, lang_code, i, data)
            all_questions[f"page_{i}"] = questions
            if i == 1 and choices:
                all_choices = choices
            time.sleep(1.5)
        except Exception as e:
            print(f"Failed at page {i} for language {lang_code}: {e}")
            break

    output = {
        "questions": all_questions,
        "choices": all_choices
    }

    output_path = os.path.join(OUTPUT_DIR, f"{lang_code}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved {lang_code} to {output_path}")

if __name__ == "__main__":
    for lang_code in tqdm(LANGUAGES.keys(), desc="Scraping languages"):
        scrape_language(lang_code)
