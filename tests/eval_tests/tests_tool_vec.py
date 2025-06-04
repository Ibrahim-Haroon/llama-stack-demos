import os
import json
import time
from llama_stack.apis.vector_io import QueryChunksResponse
import utils
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
import tools as tool

# Load environment variables
load_dotenv()

def get_tool_name(response: QueryChunksResponse):
    tool_name = response.chunks[0].content

    return tool_name

def get_tool_embedding(vector_db_id: str, client: LlamaStackClient, query: str):
    return client.vector_io.query(vector_db_id=vector_db_id, query=query)


def load_queries(file_path: str) -> List[Dict[str, str]]:
    """Load query strings from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

        if not isinstance(data, dict) or 'queries' not in data:
            raise ValueError(f"Invalid JSON format in {file_path}")

        # Return full query objects with ID for better test identification
        return data['queries']
    except FileNotFoundError:
        print(f"Query file not found: {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {file_path}")
        return []

def get_query_id(query_obj):
    """Extract an ID from a query object for better test identification."""
    if isinstance(query_obj, dict) and 'id' in query_obj:
        return query_obj['id']
    elif isinstance(query_obj, dict) and 'query' in query_obj:
        # Use first few words of query if no ID is available
        words = query_obj['query'].split()[:5]
        return '_'.join(words).lower().replace(',', '').replace('.', '')
    return "unknown_query"

def execute_query(
    client: LlamaStackClient,
    prompt: str,
    model: str,
    tools: Union[List[str], List[Any]], # list of toolgroup_ids or tool objects
    instructions: Optional[str] = None,
    max_tokens: int = 4096
) -> Dict[str, Any]:
    """Execute a single query with a given set of tools."""

    if instructions is None:
        # Default instructions for general tool use
        instructions = """
            You MUST always use a tool.
            """

    agent = Agent(
        client,
        model=model,
        instructions=instructions,
        tools=tools,
        tool_config={"tool_choice": "auto"},
        sampling_params={"max_tokens": max_tokens}
    )

    session_id = agent.create_session(session_name=f"Test_{int(time.time())}")
    print(f"Created session_id={session_id} for Agent({agent.agent_id})" if not all(isinstance(t, str) for t in tools) else "")

    turn_response = agent.create_turn(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        session_id=session_id,
        stream=False
    )
    return turn_response

def run_client_tool_test(model, vector_db_id, query_obj, llama_client, logger):
    """Run a single test for a specific server type, model, and query."""
    query_id = get_query_id(query_obj)
    prompt = query_obj['query']
    expected_tool_call = query_obj['tool_call']

    available_client_tools = {
        "add_two_numbers": tool.add_two_numbers,
        "subtract_two_numbers": tool.subtract_two_numbers,
        "multiply_two_numbers": tool.multiply_two_numbers,
        "divide_two_numbers": tool.divide_two_numbers,
        "get_current_date": tool.get_current_date,
        "greet_user": tool.greet_user,
        "string_length": tool.string_length,
        "to_uppercase": tool.to_uppercase,
        "to_lowercase": tool.to_lowercase,
        "reverse_string": tool.reverse_string,
        "is_even": tool.is_even,
        "is_odd": tool.is_odd,
        "get_max_of_two": tool.get_max_of_two,
        "get_min_of_two": tool.get_min_of_two,
        "concatenate_strings": tool.concatenate_strings,
        "is_palindrome": tool.is_palindrome,
        "calculate_square_root": tool.calculate_square_root,
        "power": tool.power,
        "get_day_of_week": tool.get_day_of_week,
        "email_validator": tool.email_validator,
        "count_words": tool.count_words,
        "average_two_numbers": tool.average_two_numbers,
        "remove_whitespace": tool.remove_whitespace,
        "convert_celsius_to_fahrenheit": tool.convert_celsius_to_fahrenheit,
        "convert_fahrenheit_to_celsius": tool.convert_fahrenheit_to_celsius,
        "convert_celsius_to_kelvin": tool.convert_celsius_to_kelvin,
        "convert_fahrenheit_to_kelvin": tool.convert_fahrenheit_to_kelvin,
        "get_substring": tool.get_substring,
        "round_number": tool.round_number,
        "is_leap_year": tool.is_leap_year,
        "generate_random_integer": tool.generate_random_integer,
        "get_file_extension": tool.get_file_extension,
        "replace_substring": tool.replace_substring,
        "is_prime": tool.is_prime,
        "calculate_bmi": tool.calculate_bmi,
        "convert_kilograms_to_pounds": tool.convert_kilograms_to_pounds,
        "convert_pounds_to_kilograms": tool.convert_pounds_to_kilograms,
        "convert_feet_to_meters": tool.convert_feet_to_meters,
        "is_alphanumeric": tool.is_alphanumeric,
        "url_encode": tool.url_encode,
        "url_decode": tool.url_decode
    }

    discovered_tool = get_tool_name(get_tool_embedding(
        query=prompt,
        vector_db_id=vector_db_id,
        client=llama_client
    ))

    tool_to_use = available_client_tools.get(discovered_tool)

    logger.info(f"Testing query '{query_id}' with model {model}")
    logger.info(f"Query: {prompt[:50]}...")

    try:
        response = execute_query(
            client=llama_client,
            prompt=prompt,
            model=model,
            tools=[tool_to_use],
        )
        # Get Tool execution and Inference steps
        steps = response.steps

        #Get tool used
        try:
            tools_used = steps[1].tool_calls[0].tool_name
        except Exception as e:
            logger.error(f"Error extracting tool name: {e}")
            tools_used = None
        tool_call_match = True if tools_used == expected_tool_call else False
        logger.info(f"Tool used: {tools_used} Tool expected: {expected_tool_call} match: {tool_call_match} ")

        #Check inference was not empty
        final_response = ""
        try:
            final_response = steps[2].api_model_response.content.strip()
            inference_not_empty = True if final_response != '' else False
        except Exception as e:
            logger.error(f"Error checking inference content: {e}")
            inference_not_empty = False
        logger.info(f'Inference not empty: {inference_not_empty}')
        logger.info(f"Query '{query_id}' succeeded with model {model} and the response \n {final_response}")

        # Record success metrics, including the expected_tool_call
        utils.add_client_tool_call_metric(
            model=model,
            query_id=query_id,
            status="SUCCESS",
            tool_call_match=tool_call_match,
            inference_not_empty=inference_not_empty,
            expected_tool_call=expected_tool_call
        )

        return True

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Query '{query_id}' failed with model {model}: {error_msg}")

        # Record failure metrics
        utils.add_client_tool_call_metric(
            model=model,
            query_id=query_id,
            status="FAILURE",
            tool_call_match=False,
            inference_not_empty=False,
            expected_tool_call=expected_tool_call,
            error=error_msg
        )

        return False

def insert_tool_embedding(tool_name, vector_db_id: str, client: LlamaStackClient):
    chunk = {
        "content": tool_name,
        "mime_type": "text/plain",
        "metadata": {
            # "tool_prompt_format": {
            # "type": "function",
            # "function" : {
            #     "name": tool_name,
            #     "description": tool_description,
            #     "parameters": tool_params
            #     }
            # },
            "document_id": tool_name
        }
    }

    client.vector_io.insert(vector_db_id=vector_db_id, chunks=[chunk])

def fill_vector_db(vector_db_id: str, client: LlamaStackClient, queries):
    query_set = set()

    for query in queries:
        tool_name = query["tool_call"]

        if tool_name not in query_set:
            insert_tool_embedding(
                tool_name=tool_name,
                vector_db_id=vector_db_id,
                client=client
            )
            query_set.add(tool_name)


def main():
    """Main function to run all tests."""
    logger = utils.setup_logger()

    base_url = os.getenv('REMOTE_BASE_URL')
    if not base_url:
        logger.error("REMOTE_BASE_URL environment variable not set")
        return
    llama_client = LlamaStackClient(base_url=base_url)

    models = ["meta-llama/Llama-3.2-3B-Instruct", ] # meta-llama/Llama-3.2-3B-Instruct, llama3.2:1b
            #  "ibm-granite/granite-3.2-8b-instruct",]
            #   "watt-ai/watt-tool-8B",
            #   "meta-llama/Llama-3.3-70B-Instruct"]
    vector_db_id = "tool_name_test_vdb" # tool_use_case_test_vdb
    emodels = llama_client.models.list()
    embedding_model = (
        em := next(m for m in emodels if m.model_type == "embedding")
    ).identifier
    embedding_dimension = 384

    _ = llama_client.vector_dbs.register(
        vector_db_id=vector_db_id,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        provider_id="faiss",
    )

    client_tool_queries = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       "queries/", "client_tool_queries.json")

    queries = load_queries(client_tool_queries)

    fill_vector_db(vector_db_id, llama_client, queries)

    total_tests = 0
    successful_tests = 0

    for model in models:
        logger.info(f"\n=== Testing with model: {model} ===\n")

        if not queries:
            logger.info(f"No queries found")
            continue

        for query_obj in queries:
            total_tests += 1
            success = run_client_tool_test(
                model,
                vector_db_id,
                query_obj,
                llama_client,
                logger
            )
            if success:
                successful_tests += 1

    # Print summary
    logger.info(f"\n=== Test Summary ===")
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Successful tests: {successful_tests}")
    logger.info(f"Failed tests: {total_tests - successful_tests}")
    if total_tests > 0:
        success_rate = (successful_tests / total_tests) * 100
        logger.info(f"Success rate: {success_rate:.1f}%")

    # Generate plots
    logger.info(f"\n=== Generating plots ===")
    utils.get_analysis_plots(per_tool_plot=True)


if __name__ == "__main__":
    main()
