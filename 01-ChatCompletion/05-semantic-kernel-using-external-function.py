from dotenv import load_dotenv
import asyncio
import os
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions import KernelArguments
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents.chat_history import ChatHistory
from typing import Annotated
from semantic_kernel.functions import kernel_function
from plugins.ExternalPlugin.JobSearch.JobSearch import JobSearchPlugin

# Load environment variables from .env file
load_dotenv()

# Get the plugin directory
current_dir = os.path.abspath("")
plugin_dir = os.path.join(current_dir, "plugins")

# Get Azure OpenAI credentials from environment
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
serp_api_key = os.getenv("SERP_API_KEY")


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
    plugin = kernel.add_plugin(JobSearchPlugin(api_key=serp_api_key), plugin_name="search_jobs")
    job_function=plugin["SearchJobs"]
    
    #print(job_function)    
    job = await kernel.invoke(
        job_function,
        KernelArguments(query="Software Engineers"),
    )
    print(job)

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())