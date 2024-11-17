# This file contains the code to paraphrase a message
import openai

def paraphrase_message(client, message):
    # Prepare the prompt from the conversation history
    prompt = message

    try:
        # Generate a response using GPT-4o mini
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Replace with the correct model name if needed
            messages=[
                {"role": "system", "content": "You are an assistant tasked with paraphrasing text. For each input, your goal is to rephrase it using different wording and sentence structures while ensuring that the original meaning, intent, and nuances are completely preserved. Do not omit or add new information. The paraphrased output should be clear, concise, and faithful to the original message."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,  # Adjust as needed
            temperature=0.0,  # Adjust as needed for creativity
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

if __name__ == "__main__":

    # Example usage:
    message = "What are you up to today?"

    client = openai.OpenAI()
    response = paraphrase_message(client, message)
    print("Generated response:", response)
