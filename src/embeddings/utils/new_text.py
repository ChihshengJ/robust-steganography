# This file contains the code to sample a new message
import random
import re

import openai

from ..config.constants import STEGO_GEN_MODEL


def clean_response(text):
    # Regex to find the last full sentence ending with ., !, or ?
    match = re.search(r"([.!?])[^.!?]*$", text)
    if match:
        return text[: match.end()].strip()
    else:
        return text.strip()


def generate_response(
    client,
    conversation_history,
    system_prompt="You are a highly dynamic conversational model tasked with generating responses that are extremely varied in tone, content, and structure.",
    max_length=300,
    temperature=1.0,
    top_p=1.0,
    decomp_mode=False,
):
    # Prepare the prompt from the conversation history
    # adding datetime noise to disable prompt caching
    if not decomp_mode:
        prompt = "\n".join(conversation_history) + "\n"
    else:
        prompt = conversation_history
    # print(f"prompt: {prompt}")

    format = {"type": "json_object"} if decomp_mode else None

    try:
        # Generate a response using GPT-4o mini
        response = client.chat.completions.create(
            model=STEGO_GEN_MODEL,  # Original model name preserved
            response_format=format,
            messages=[
                {
                    "role": "system",
                    "content": f"{random.uniform(1.0, 10000.0)}\n" + system_prompt,
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
        if isinstance(conversation_history, list):
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
    response = generate_response(client, conversation_history)
    print("Generated response:", response)
