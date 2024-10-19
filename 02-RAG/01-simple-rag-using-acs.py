import os
import json
from dotenv import load_dotenv

# Add OpenAI import
from openai import AzureOpenAI


def main():

    try:
        # Flag to show citations
        show_citations = True

        # Get configuration settings
        load_dotenv()
        azure_oai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_oai_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_oai_chat_completion_deployment = os.getenv(
            "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
        azure_oai_version = os.getenv("AZURE_OPENAI_API_VERSION")
        azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        azure_search_key = os.getenv("AZURE_SEARCH_KEY")
        azure_search_index = os.getenv("AZURE_SEARCH_INDEX")

        # Initialize the Azure OpenAI client
        client = AzureOpenAI(
            # base_url=f"{azure_oai_endpoint}openai/deployments/{azure_oai_chat_completion_deployment}/extensions",
            azure_endpoint=azure_oai_endpoint,
            api_key=azure_oai_key,
            api_version=azure_oai_version)

        # Get the prompt
        text = input('\nEnter a question:\n')

        extra_body = {
            "data_sources": [
                {
                    "type": "azure_search",
                    "parameters": {
                        "endpoint": azure_search_endpoint,
                        "index_name": azure_search_index,
                        "authentication": {
                            "type": "api_key",
                            "key": azure_search_key
                        }
                    }
                }
            ]
        }

        # Send request to Azure OpenAI model
        print("...Sending the following request to Azure OpenAI endpoint...")
        print("Request: " + text + "\n")

        response = client.chat.completions.create(
            model=azure_oai_chat_completion_deployment,
            messages=[
                {"role": "system", "content": "You are a helpful travel agent"},
                {"role": "user", "content": text}
            ],
            extra_body=extra_body
        )

        # Print response
        print("Response: " + response.choices[0].message.content + "\n")

        if (show_citations):
            # Print citations
            print("Citations:")
            citations = response.choices[0].message.context["citations"]
            for c in citations:
                print("  Title: " + c['title'])
                # print("  Title: " + c['title'] + "\n    URL: " + c['url'])

    except Exception as ex:
        print(ex)


if __name__ == '__main__':
    main()
