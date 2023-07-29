# FILE: Data processing


# DEPENDENCIES
import shared as s  # Shared values
import requests
import json
import numpy as np


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
              account_id: str,
              key: str = s.API_KEY,
              base: str = s.API_STREAM_BASE_URL,
              ticker: str = s.CURRENCY,
              verbose: bool = True):

    # FUNCTION: Subscribe to a channel and get a certain amount of raw data

    # PARAM (required): iterations: int: How many data points to collect.
    # PARAM (required): account_id: str: Account ID
    # PARAM: key: str: API key
    # PARAM: base: str: API stream base URL
    # PARAM: ticker: str: Currency pair to track eg. "EUR_USD"
    # PARAM: verbose: Whether to print logs

    # RETURN: data: ndarray: 2D numpy array of data points. Each point is formatted like: [close_bid, close_ask]

    s.log(tag="api",
          content=f"Subscribing to {ticker} stream for {iterations} iterations...",
          verbose=verbose)  # Log

    headers = {
        "Authorization": f"Bearer {key}"
    }  # Headers

    params = {
        "instruments": ticker
    }  # Query parameters

    path = f"/v3/accounts/{account_id}/pricing/stream"  # Path for API call
    endpoint = base + path  # Full endpoint URL

    response = requests.get(url=endpoint, headers=headers, params=params, stream=True)  # Subscribe

    if response.status_code == 200:  # If connection is successful

        s.log(tag="api",
              content=f"WebSocket connection successful...",
              verbose=verbose)  # Log

        iteration = 0  # Iteration counter
        data = []  # Dataset
        for point_raw in response.iter_lines():  # Iterate over response lines
            if point_raw and iteration < iterations:  # If line received

                point_str = point_raw.decode()  # Decode bytes to str
                point = json.loads(point_str)  # Parse str to dict

                if point["type"] == "PRICE":  # If given data is price

                    # bids = point["bids"]  # Bid price (sell at this price)  --> Currently not in use
                    # asks = point["asks"]  # Ask price (buy at this price)  --> Currently not in use
                    close_bid = float(point["closeoutBid"])  # Price to close a long (buy) position, i.e. to sell assets
                    close_ask = float(point["closeoutAsk"])  # Price to close a long (buy) position, i.e. to buy assets

                    data.append([close_bid, close_ask])  # Append data point

                    s.log(tag="api",
                          content=f"Data point {iteration + 1} of {iterations} received.",
                          verbose=verbose)  # Log

                    iteration += 1  # Increment iteration counter

                else:
                    s.log(tag="api",
                          content=f"Extraneous data received & dropped.",
                          verbose=verbose)  # Log

            elif iteration >= iterations:  # If loop is over

                data = np.array(data)

                s.log(tag="api",
                      content=f"Data collection of {iterations} points complete.",
                      verbose=verbose)  # Log

                return data  # Return result and exit


def buy(account_id: str,
        units: int,
        key: str = s.API_KEY,
        base: str = s.API_REST_BASE_URL,
        ticker: str = s.CURRENCY,
        verbose: bool = True):

    # FUNCTION: Buy base currency and sell quote currency (e.g. for EUR_USD, buy EUR and pay USD)

    # PARAM (required): account_id: str: Account ID
    # PARAM (required): units: int: Units of base currency to buy (e.g. for EUR_USD, EUR)
    # PARAM: key: str: API key
    # PARAM: base: str: API REST base URL
    # PARAM: ticker: str: Currency pair to buy
    # PARAM: verbose: bool: Whether to print logs

    # RETURN (buy order): Buys forex

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }  # Headers

    request = {
        "order": {
            "units": f"{units}",
            "instrument": f"{ticker}",
            "timeInForce": "FOK",  # Fill Or Kill
            "type": "MARKET",
            "positionFill": "DEFAULT"
        }
    }
    request = json.dumps(request)

    path = f"/v3/accounts/{account_id}/orders"  # API path

    endpoint = base + path  # Full API endpoint URL

    response = requests.post(url=endpoint, headers=headers, data=request)  # Make the request

    if response.status_code == 201:  # If request is successful

        response_str = response.text  # Response content as str
        response_dict = json.loads(response_str)  # Response content as dict
        s.log(tag="api",
              content=f"Placed order for {units} units of {ticker}.",
              verbose=verbose)  # Log
        # Bear in mind if the order is not fulfilled immediately, it is killed but still returns 201.
        try:
            cancel_status = response_dict["orderCancelTransaction"]["type"]  # Status of cancellation
            cancel_reason = response_dict["orderCancelTransaction"]["reason"]  # Reason for cancellation
            if cancel_status == "ORDER_CANCEL":  # If order gets cancelled
                s.log(tag="api",
                      content=f"Order cancelled. Error: {cancel_reason}",
                      verbose=verbose)  # Log

        except:
            pass

    else:

        s.log(tag="api",
              content=f"Failed to buy {ticker}. Error: {response.text}",
              verbose=verbose)  # Log


def sell(account_id: str,
         units: int,
         key: str = s.API_KEY,
         base: str = s.API_REST_BASE_URL,
         ticker: str = s.CURRENCY,
         verbose: bool = True):

    # FUNCTION: Buy base currency and sell quote currency (e.g. for EUR_USD, buy EUR and pay USD)

    # PARAM (required): account_id: str: Account ID
    # PARAM (required): units: int: Units of base currency to buy (e.g. for EUR_USD, EUR)
    # PARAM: key: str: API key
    # PARAM: base: str: API REST base URL
    # PARAM: ticker: str: Currency pair to buy
    # PARAM: verbose: bool: Whether to print logs

    # RETURN (sell order): Sells forex.

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }  # Headers

    request = {
        "order": {
            "units": f"{-1 * units}",  # Times -1 to sell.
            "instrument": f"{ticker}",
            "timeInForce": "FOK",  # Fill Or Kill
            "type": "MARKET",
            "positionFill": "DEFAULT"
        }
    }
    request = json.dumps(request)

    path = f"/v3/accounts/{account_id}/orders"  # API path

    endpoint = base + path  # Full API endpoint URL

    response = requests.post(url=endpoint, headers=headers, data=request)  # Make the request

    if response.status_code == 201:  # If request is successful

        response_str = response.text  # Response content as str
        response_dict = json.loads(response_str)  # Response content as dict
        s.log(tag="api",
              content=f"Placed order to sell {units} units of {ticker}.",
              verbose=verbose)  # Log
        # Bear in mind if the order is not fulfilled immediately, it is killed but still returns 201.
        try:
            cancel_status = response_dict["orderCancelTransaction"]["type"]  # Status of cancellation
            cancel_reason = response_dict["orderCancelTransaction"]["reason"]  # Reason for cancellation
            if cancel_status == "ORDER_CANCEL":  # If order gets cancelled
                s.log(tag="api",
                      content=f"Order cancelled. Error: {cancel_reason}",
                      verbose=verbose)  # Log

        except:
            pass

    else:

        s.log(tag="api",
              content=f"Failed to sell {ticker}. Error: {response.text}",
              verbose=verbose)  # Log


# MAIN
s.account_id = accounts()[0]["id"]  # Set account id - KEEP IN CODE


# TESTING
#sell(account_id=s.account_id, units=1)
