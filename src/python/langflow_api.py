import json
import logging
import requests
import os
import concurrent.futures
from typing import Optional, Dict, Any, List, Tuple
from tenacity import retry, wait_exponential, stop_after_attempt


logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    reraise=True,
)
def upload_file(
    langflow_base_url: str,
    flow_id: str,
    file_path: str,
    api_key: Optional[str] = None,
) -> str:
    """
    Upload a file to Langflow and return the server file path.
    """
    api_url = f"{langflow_base_url}/api/v1/files/upload/{flow_id}"
    
    headers = {"x-api-key": api_key} if api_key else None
    
    with open(file_path, 'rb') as file:
        files = {'file': file}
        response = requests.post(api_url, headers=headers, files=files)
        response.raise_for_status()
        
        # Response should contain the server file path
        upload_data = response.json()
        return upload_data.get('file_path')


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    reraise=True
)
def run_flow(
    langflow_base_url: str,
    flow_id: str,
    file_path: str,
    tweaks: Optional[Dict[str, Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
) -> dict:
    """
    Run a flow with retries and exponential backoff.
    
    Args:
        file_path: Path to the file to upload
        tweaks: Dictionary of tweaks to apply to the flow, defaults to TWEAKS global variable
        api_key: Optional API key for authentication
        
    Returns:
        The JSON response from the flow
    """
    
    # First upload the file
    server_file_path = upload_file(file_path, api_key)
    logger.info(f"Server file path: {server_file_path}")
    
    api_url = f"{langflow_base_url}/api/v1/run/{flow_id}"
    
    # Update the file path in the tweaks with the server path
    for component_id, _ in tweaks.items():
        if component_id.startswith("File-") or component_id.startswith("NVIDIAIngest-"):
            tweaks[component_id]["path"] = server_file_path
    
    payload = {
        "input_value": "",
        "tweaks": tweaks,
        "output_type": "text",
        "input_type": "text",
    }

    logger.debug(f"Payload: {json.dumps(payload, indent=2)}")
    
    headers = {
        "Content-Type": "application/json",
        **({"x-api-key": api_key} if api_key else {})
    }
    
    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=120)
        logger.info(f"Status code: {response.status_code}")
        logger.debug(f"Response headers: {response.headers}")
        logger.debug(f"Response text: {response.text}")
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error: {str(e)}")
        logger.error(f"Response content: {e.response.text}")
        raise
    except requests.exceptions.Timeout:
        logger.error("Request timed out after 2 minutes")
        raise


def process_file(
    file_path: str,
    custom_tweaks: Dict[str, Dict[str, Any]],
    api_key: Optional[str] = None,
) -> Tuple[str, dict, Optional[Exception]]:
    """
    Process a single file and return the result along with any exceptions.
    
    Args:
        file_path: Path to the file to process
        custom_tweaks: Optional custom tweaks to use instead of defaults
        api_key: Optional API key
        
    Returns:
        Tuple of (file_path, result, exception)
        If successful, exception will be None
        If failed, result will be an empty dict and exception will contain the error
    """
    try:
        result = run_flow(file_path, tweaks=custom_tweaks, api_key=api_key)
        return (file_path, result, None)
    except Exception as e:
        return (file_path, {}, e)


def process_files_in_parallel(
    file_paths: List[str], 
    custom_tweaks: Dict[str, Dict[str, Any]],
    max_workers: int = 4,
    api_key: Optional[str] = None,
) -> List[Dict]:
    """
    Process multiple files in parallel and return results as a list of dictionaries.
    
    Args:
        file_paths: List of file paths to process
        max_workers: Maximum number of parallel workers
        custom_tweaks: Optional custom tweaks to apply to each file
        api_key: Optional API key
        
    Returns:
        List of dictionaries with results and status for each file
    """
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(
                process_file,
                file_path,
                custom_tweaks,
                api_key,
            ): file_path 
            for file_path in file_paths
        }
        
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                file_path, result, exception = future.result()
                status = "Success" if exception is None else f"Failed: {str(exception)}"
                results.append({
                    "file_path": file_path,
                    "file_name": os.path.basename(file_path),
                    "status": status,
                    "result": result,
                    "exception": exception,
                })
            except Exception as e:
                results.append({
                    "file_path": file_path,
                    "file_name": os.path.basename(file_path),
                    "status": f"Error: {str(e)}",
                    "result": {},
                    "exception": e,
                })
    
    return results