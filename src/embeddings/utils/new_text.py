# This file contains the code to sample a new message
import random
import re

import openai

from ..config.constants import STEGO_GEN_MODEL


def clean_response(text) -> str:
    # Regex to find the last full sentence ending with ., !, or ?
    match = re.search(r"([.!?])[^.!?]*$", text)
    if match:
        return text[: match.end()].strip()
    else:
        return text.strip()


def generate_response(
    client,
    prompt: str | list[str],
    system_prompt: str,
    max_length: int = 300,
    temperature: float = 1.0,
    top_p: float = 1.0,
    json_mode: bool = False,
) -> str:
    # Prepare the prompt from the conversation history
    # adding datetime noise to disable prompt caching
    if not json_mode:
        prompt = "\n".join(prompt) + "\n"

    format = {"type": "json_object"} if json_mode else None

    try:
        # Generate a response using GPT-4o mini
        response = client.chat.completions.create(
            model=STEGO_GEN_MODEL,  # Original model name preserved
            response_format=format,
            messages=[
                {
                    "role": "system",
                    "content": f"{random.uniform(1.0, 100000.0)}\n" + system_prompt,
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_length,  # Use passed in max_length
            temperature=temperature,  # Preserved original temperature
            top_p=top_p,
            # stop=["\n"],
        )
        # print("initial response:\n", response)

        # Extract and return the generated response text
        text = response.choices[0].message.content.strip()
        if isinstance(prompt, list):
            text = clean_response(text)
        return text

    except Exception as e:
        return f"An error occurred: {e}"


if __name__ == "__main__":
    # Example usage:
    conversation_history = [
        "What are you up to today?",
        "Nothing much, I'm just working on a project.",
        "Do you want me to take a look? We can grab some coffee.",
    ]

    client = openai.OpenAI()
    response = generate_response(client, conversation_history, system_prompt="")
    print("Generated response:", response)
