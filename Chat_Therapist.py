import requests

# Function to interact with the OpenAI API
def chat_with_therapist(user_input):
    api_key = ''  # Replace with your OpenAI API key
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    prompt = f"As a supportive therapist, respond empathetically to: {user_input}"
    data = {
        'model': 'gpt-3.5-turbo',  # or another model you prefer
        'messages': [{'role': 'user', 'content': prompt}]
    }

    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)

    # Print the response for debugging
    print(response.status_code)  # Print the HTTP status code
    print(response.json())        # Print the full JSON response

    return response.json()['choices'][0]['message']['content']


# Main interaction loop
def main():
    print("Welcome to the Chat Therapist. Share your thoughts below:")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Therapist: Take care! Goodbye.")
            break
        response = chat_with_therapist(user_input)
        print(f"Therapist: {response}")

if __name__ == "__main__":
    main()
