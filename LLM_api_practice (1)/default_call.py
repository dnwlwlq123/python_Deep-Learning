from pprint import pprint
from openai import OpenAI
from logging_util import log_chatgpt_call

openai_api_key = ''
client = OpenAI(api_key=openai_api_key)

@log_chatgpt_call
def ask_chatgpt(query: str, model: str = 'gpt-4o-mini-2024-07-18'):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Translate the following text to English: {query}"}
        ]
    )
    return completion

def ask_chatgpt_reverse(query: str, model: str = 'gpt-4o-mini-2024-07-18'):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Translate the following text to Korean: {query}"}
        ]
    )
    return completion

def render_result(completion):
    d = completion.to_dict()
    result = d['choices'][0]['message']['content']

    import re
    kor_text = re.findall('[가-힣]+', result)
    if kor_text:
        return " ".join(kor_text)
    return result.strip()

def translate_to_english(text: str):
    completion = ask_chatgpt(text)
    translated_text = render_result(completion)
    return translated_text

def translate_to_korean(text: str):
    completion = ask_chatgpt_reverse(text)
    translated_text = render_result(completion)
    return translated_text

if __name__ == '__main__':
    while True:
        mode = input("선택 (1: 한국어 -> 엉여, 2: 영어 -> 한국어 q: 종료): ")
        if mode == '1':
            korean_text = input("한국어를 입력하세요: ")
            translated_text = translate_to_english(korean_text)
            print(f"번역된 영어: {translated_text}")
        elif mode == '2':
            english_text = input("영어를 입력하세요: ")
            translated_text = translate_to_korean(english_text)
            print(f"번역된 한국어: {translated_text}")
        elif mode == 'q':
            break
        else:
            print("잘못된 입력입니다. 다시 시도하세요.")
