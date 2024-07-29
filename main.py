import glob
import pandas as pd
from tqdm import tqdm
import pathlib
from cleantext import clean
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from google.cloud import translate_v2 as translate


translate_client = translate.Client(client_options={
    'api_key': 'AIzaSyD85YE2TjUeRi15cHrd4VtqTRGv5EeXWpE'
})

def clean_text(text):
    return clean(
    text = text,
    fix_unicode=True,
    to_ascii=False,
    lower=False,
    no_urls=False,
    no_emails=False,
    no_phone_numbers=False,
    no_numbers=False,
    no_digits=False,
    no_currency_symbols=False,
    no_punct=False,
    lang="en"
).replace('""', '"').replace("'", '`')

def translate_text(text: str) -> dict:
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """


    if isinstance(text, bytes):
        text = text.decode("utf-8")

    result = translate_client.translate(text, source_language='en', target_language='uz')

    return result



def process_row(row):
    sentences_instruction = [clean_text(sentence) + "." for sentence in str(row['instruction']).split(".") if sentence.strip()]
    sentences_context = [clean_text(sentence) + "." for sentence in str(row['input']).split(".") if sentence.strip()]
    sentences_response = [clean_text(sentence) + "." for sentence in str(row['output']).split(".") if sentence.strip()]

    sentences_instruction = " ".join(sentences_instruction)
    sentences_context = " ".join(sentences_context)
    sentences_response = " ".join(sentences_response)

    translated_instruction = translate_text(sentences_instruction)
    translated_context = translate_text(sentences_context)
    translated_response = translate_text(sentences_response)

    return {'instruction': clean_text(translated_instruction['translatedText']), 'input': clean_text(translated_context['translatedText']), 'output': clean_text(translated_response['translatedText'])}

def save_to_csv(content, file_name, save_row, total_count):
    pathlib.Path(f'./Instruction_tuning_dataset3_/{file_name}').mkdir(exist_ok=True, parents=True)
    df_out = pd.DataFrame(data=content)
    df_out.to_csv(f'./Instruction_tuning_dataset3_/{file_name}/{file_name}-{save_row//100}-{total_count}.csv', index=False)

def main():
    ds = load_dataset("ldbb123/Instruction-tuning_Datasets", split=f"train[83000:100000]")
    df = pd.DataFrame(ds)
    file_name = 'Instruction_tuning_dataset'
    total_count = len(df) // 100
    save_row = 1
    content = {'instruction': [], 'input': [],'output': []}


    with ThreadPoolExecutor(max_workers=50) as executor:
        future_to_row = {executor.submit(process_row, row): row for _, row in df.iterrows()}
        for future in tqdm(as_completed(future_to_row), total=len(future_to_row)):
            result = future.result()
            if result:
                content['instruction'].append(result['instruction'])
                content['input'].append(result['input'])
                content['output'].append(result['output'])

                if save_row % 100 == 0:   # CHANGE this PART
                    print(content['instruction'][-1:], content['input'][-1:],content['output'][-1:])
                    save_to_csv(content, file_name, save_row, total_count)
                    content = {'instruction': [], 'input': [], 'output': []}
            save_row += 1

    if content['output']:
        save_to_csv(content, file_name, save_row-1, total_count)

if __name__ == "__main__":
    main()
