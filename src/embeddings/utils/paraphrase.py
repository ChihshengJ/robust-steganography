# This file contains the code to paraphrase a message
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


def paraphrase_message(client, message):
    # Prepare the prompt from the conversation history
    prompt = message

    try:
        # Generate a response using GPT-4o mini
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Replace with the correct model name if needed
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant tasked with paraphrasing text. For each input, your goal is to rephrase it using different wording and sentence structures while ensuring that the original meaning, intent, and nuances are completely preserved. Do not omit or add new information. The paraphrased output should be clear, concise, and faithful to the original message.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=100,  # Adjust as needed
            temperature=0.5,  # Adjust as needed for creativity
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            # stop=["\n"],
        )
        # Extract and return the generated response text
        text = response.choices[0].message.content.strip()
        return text

    except Exception as e:
        return f"An error occurred: {e}"


def paraphrase_message_ollama(host, message):
    print(f"original message:{message}\n----------split------------")
    data = {
        "model": "qwen3:8b",  # Replace with the correct model name if needed
        "system": """
            You are an assistant tasked with paraphrasing text. For each input, your goal is to rephrase 
            it using different wording and sentence structures while ensuring that the original meaning, 
            intent, and nuances are completely preserved. Do not omit or add new information.
            The paraphrased output should be clear, concise, and faithful to the original message.
            Do not change the tone or conversational style of the original text. Do not make the paraphrase
            longer than the original text. Do not add new lines. No emojis allowed.""",
        "messages": [
            {"role": "user", "content": message},
        ],
        "options": {
            "num_predict": 100,  # Adjust as needed
            "temperature": 0.3,  # Adjust as needed for creativity
        },
        "think": False,
        "stream": False,
    }
    retry = Retry(
        total=5,
        backoff_factor=20,
        status_forcelist=[429, 500, 529],  # Only retry on rate limits and server errors
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)

    try:
        response = session.post(host, json=data)

        if response.status_code == 200:
            print(response.json()['message']['content'])
            print("-------------split---------------")
            return str(response.json()["message"]["content"]).strip()

        error_msg = response.json()
        print(f"Error Details: {error_msg}")
        return None

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

    finally:
        time.sleep(0.25)


if __name__ == "__main__":
    # Example usage:
    message = "Why is the sky blue"

    # client = openai.OpenAI()
    response = paraphrase_message_ollama("http://192.168.2.34:11434/api/chat", message)
    print("Generated response:", response)
