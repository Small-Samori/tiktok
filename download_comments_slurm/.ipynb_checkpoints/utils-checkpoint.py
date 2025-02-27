import requests
import dotenv
import os, time
import pandas as pd
import glob, json

dotenv.load_dotenv()

##### ------ #####
def get_access_token(client_key, client_secret):
    
    endpoint_url = "https://open.tiktokapis.com/v2/oauth/token/"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
    }

    data = {
        'client_key': client_key,
        'client_secret': client_secret,
        'grant_type': 'client_credentials'
    }

    response = requests.post(endpoint_url, headers=headers, data=data)

    if response.status_code == 200:
        response_json = response.json()
        return response_json
        
    else:
        # If the request was not successful, print the error response JSON
        print("Error:", response.json())

##### ------ #####
def query_comments_api(query_body, query_params, headers, save_folder):
    endpoint_name = "comments"
    endpoint = "https://open.tiktokapis.com/v2/research/video/comment/list/"
    
    video_id = query_body["video_id"]
    prev_cursor = query_body["cursor"]
    
    # make post request
    response = requests.post(endpoint, json=query_body, params=query_params, headers=headers)
    status_code = response.status_code

    if response.status_code != 200:
        try:
            response_json = response.json()
        except json.JSONDecodeError as e:
            response_json = "Error reading json"
            
        return video_id, False, prev_cursor, response_json, status_code 

    
    # extracting information for pagination
    data = response.json().get("data", {})
    
    has_more = data["has_more"]
    cursor = data["cursor"]

    # saving queried data
    records = data.get(endpoint_name, [])
    df = pd.DataFrame(records)
    if len(df) != 0:
        df.to_csv(f"{save_folder}/{endpoint_name}_{video_id}_{cursor}.csv", index=False)

    return video_id, has_more, cursor, len(df), status_code

##### ------ #####
def query_comments_api_paginate(fields, cursor, video_id, save_folder, log_name, credentials, log_dir="./logs_download_comments_slurm"):
    
    access_token = credentials["access_token"]
    token_type = credentials["token_type"]
    
    query_params = {"fields": fields}
    query_body = {
        "video_id":video_id, 
        "max_count":100, "cursor":cursor
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"{token_type} {access_token}"
    }
    
    has_more = True

    # pagination loop
    while has_more:
        query_body.update({"cursor": cursor})
            
        video_id, has_more, cursor, samples, status_code = query_comments_api(query_body, query_params, headers, save_folder)

        with open(f"{log_dir}/{log_name}", "a") as f:
            f.write(f"{cursor}\t{has_more}\t{samples}\t{video_id}\t{status_code}\n")

        # print(f"{cursor}\t{has_more}\t{samples}\t{video_id}\t{status_code}")
        time.sleep(5)
        
    return status_code

##### ------ #####
def get_comments(video_ids, fields, cursor, save_folder, log_dir="./logs_download_comments_slurm"):

    client_key = os.getenv("CLIENT_KEY")
    client_secret = os.getenv("CLIENT_SECRET")
    
    credentials = get_access_token(client_key, client_secret)

    logs = glob.glob(f'{log_dir}/*.txt')
    log_name = f"comments_slurm_download_{len(logs)+1}.txt"

    col_names = ["cursor", "has_more","samples","video_id","status_code"]
    downloaded_vid_ids = []
    for log in logs:
        log_data = pd.read_csv(log, sep='\t', names=col_names, header=None)
        log_ids = list(log_data['video_id'])
        log_ids = [str(i) for i in log_ids]
        
        downloaded_vid_ids += log_ids
    downloaded_vid_ids = list(set(downloaded_vid_ids))

    status_code = 200
    # i = 0
    for video_id in video_ids:
        if str(video_id) not in downloaded_vid_ids:
            if status_code != 429:
                if status_code == 401:
                    credentials = get_access_token(client_key, client_secret)
                    status_code = query_comments_api_paginate(fields, cursor, video_id, save_folder, log_name, credentials)
                else:
                    status_code = query_comments_api_paginate(fields, cursor, video_id, save_folder, log_name, credentials)
                # i += 1
            else:
                break
        
        # if i == 5:
        #     break