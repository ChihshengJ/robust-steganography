# This file contains the code to sample a new message

import re

import openai


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
    max_length=100,
):
    # Prepare the prompt from the conversation history
    prompt = "\n".join(conversation_history) + "\n"

    try:
        # Generate a response using GPT-4o mini
        response = client.chat.completions.create(
            model="gpt-4.1-nano",  # Original model name preserved
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_length,  # Use passed in max_length
            temperature=1.0,  # Preserved original temperature
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            # stop=["\n"],
        )

        # Extract and return the generated response text
        text = response.choices[0].message.content.strip()
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
