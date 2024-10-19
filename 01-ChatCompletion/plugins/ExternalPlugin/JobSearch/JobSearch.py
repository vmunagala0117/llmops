import os
from dotenv import load_dotenv
from serpapi import GoogleSearch
from typing import Annotated
from semantic_kernel.functions import kernel_function

# Ref: https://serpapi.com/google-jobs-api

class JobSearchPlugin:
    # optionally pass the api_key directly or set it in the environment
    # variables
    def __init__(self, api_key: str = None):        
        # get environment variable SERPAPI_API_KEY if not passed
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY")

    @kernel_function(
        name="SearchJobs",
        description="Searches for jobs using Google Jobs API",
    )
    def search_jobs(self,
                    query: Annotated[str,
                                     "The input search query, e.g., 'Java Programmer'"]) -> Annotated[str,
                                                                                                      "The output"]:
        params = {
            "engine": "google_jobs",
            "q": query,
            "hl": "en",
            'gl': 'us',
            "api_key": self.api_key
        }

        search = GoogleSearch(params)
        jobs_results = search.get_dict()      

        if not jobs_results:
            return "No job results found."
        
        jobs_results = jobs_results["jobs_results"]

        # Process and format the results for output
        formatted_results = [
            f"{job['title']} at {job['company_name']}" for job in jobs_results]
        return '\n'.join(formatted_results)
