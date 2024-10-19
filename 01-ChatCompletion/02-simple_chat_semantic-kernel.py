import asyncio
import os
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.prompt_template import PromptTemplateConfig

# Load environment variables from .env file
load_dotenv()

# Get Azure OpenAI credentials from environment
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

print(deployment_name)
print(endpoint)

# Initialize the Semantic Kernel
kernel = Kernel()

# Add AzureChatCompletion to the kernel object
service_id = "chat_completion"
azure_chat_service = AzureChatCompletion(
    deployment_name=deployment_name,
    api_key=api_key,
    endpoint=endpoint,
    api_version="2024-02-15-preview",
    service_id=service_id
)
kernel.add_service(azure_chat_service)

# Specify Prompt Settings
prompt_settings = kernel.get_prompt_execution_settings_from_service_id(
    service_id)
print(prompt_settings)

# Create Prompt Function
prompt_text = "Explain the benefits of using Semantic Kernel in AI development."
prompt_template = PromptTemplateConfig(
    name="Prompt_Template_Configuration",
    template=prompt_text,
    template_format="semantic-kernel",
    execution_settings=prompt_settings
)
print(prompt_template)

#invoke the function
prompt_function = kernel.add_function(plugin_name="simple_prompt_plugin",
                                      prompt_template_config=prompt_template,
                                      function_name="simple_prompt_function"
                                      )

# Use the kernel to interact with the chat completion service


async def run_chat():
    response = await kernel.invoke(function=prompt_function)
    print("Response from Azure OpenAI:")
    print(response)

# Execute the function
asyncio.run(run_chat())
