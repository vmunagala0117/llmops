from dotenv import load_dotenv
import asyncio
import os
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions import KernelArguments

# Load environment variables from .env file
load_dotenv()

# Get the plugin directory
current_dir = os.path.abspath("")
plugin_dir = os.path.join(current_dir, "plugins")

# Get Azure OpenAI credentials from environment
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")


async def main():
    # Check for missing environment variables
    if not deployment_name or not endpoint or not api_key:
        raise ValueError(
            "Missing necessary Azure OpenAI environment variables")

    # Initialize the kernel
    kernel = Kernel()
    kernel.remove_all_services()
    
    service_id = "default"
    kernel.add_service(
        AzureChatCompletion(
            deployment_name=deployment_name,
            api_key=api_key,
            endpoint=endpoint,
            api_version="2024-02-15-preview",
            service_id=service_id
        ),
    )

    # Load a Plugin and run a semantic function:
    plugin = kernel.add_plugin(
        parent_directory=plugin_dir,
        plugin_name="FunPlugin")

    joke_function = plugin["Joke"]
    joke = await kernel.invoke(
        joke_function,
        KernelArguments(input="time travel to dinosaur age", style="super"),
    )
    print(joke)

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
