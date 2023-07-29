# FILE: Data processing


# DEPENDENCIES
import shared as s  # Shared values
import requests
import json
import websockets
import asyncio



# FUNCTIONS
def accounts(key: str = s.API_KEY,
             base: str = s.API_REST_BASE_URL,
             verbose: bool = True):

    # FUNCTION: Get list of accounts

    # PARAM: key: str: API key
    # PARAM: base: str: Base url of REST API
    # PARAM: verbose: Whether to print logs

    # RETURN: accounts_list: list[dict]: Accounts found for given API key

    s.log(tag="api",
          content="Getting accounts...",
          verbose=verbose)  # Log

    headers = {
        "Authorization": f"Bearer {key}"
    }  # Headers

    path = "/v3/accounts"  # Path for API call

    endpoint = base + path  # Full endpoint URL

    response = requests.get(url=endpoint, headers=headers)  # Make the request

    if response.status_code == 200:  # If response is ok

        response_str = response.text  # Response content as str
        response_dict = json.loads(response_str)  # Response content as dict
        accounts_list = response_dict["accounts"]  # List of accounts
        s.log(tag="api",
              content=f"Got {len(accounts_list)} account(s).",
              verbose=verbose)  # Log
        return accounts_list  # Return list of found accounts

    else:  # If response is not ok

        response_str = response.text  # Response content as str
        s.log(tag="api",
              content=f"Failed to get accounts. Error: {response_str}",
              verbose=verbose)  # Log


def subscribe(iterations: int,
              key: str = s.API_KEY,
              base: str = s.API_STREAM_BASE_URL,
                    ticker: str = s.CURRENCY,
                    account_id: str = s.account_id,
                    verbose: bool = True):

    # FUNCTION: Subscribe to a channel and get a certain amount of raw data

    # PARAM (required): iterations: int: How many data points to collect.
    # PARAM: key: str: API key
    # PARAM: base: str: API streaming base URL
    # PARAM: ticker: str: Currency pair to track eg. "EUR_USD"
    # PARAM: account_id: str: Account ID
    # PARAM: verbose: Whether to print logs

    # TODO RETURN:

    s.log(tag="api",
          content=f"Subscribing to {ticker} stream for {iterations} iterations...",
          verbose=verbose)  # Log

    headers = {
        "Authorization": f"Bearer {key}"
    }  # Headers

    query = f"?instruments={ticker}&snapshot=False"  # Query portion of URL
    path = f"/v3/accounts/{account_id}/pricing/stream"  # Path for API call
    endpoint = base + path + query  # Full endpoint URL

    response = requests.get(url=endpoint, headers=headers)  # Make the request

    print(response.text)


# MAIN
s.account_id = accounts()[0]["id"]  # Set account id

subscribe(iterations=10)
