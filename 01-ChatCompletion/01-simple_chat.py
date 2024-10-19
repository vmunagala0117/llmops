import os
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)


def chat(user_message):
    completion = client.chat.completions.create(
        model="gpt-4",  # model = "deployment_name".
        messages=[
            {"role": "system",
             "content": "Assistant is a large language model trained by OpenAI."},
            {"role": "user",
                "content": user_message}
        ])
    return completion

# print(response)
# print(response.model_dump_json(indent=2))
# print(response.choices[0].message.content)


print("How can I help you today?")

while True:
    user_message = input(">> ")
    response = chat(user_message)
    print(response.choices[0].message.content)
