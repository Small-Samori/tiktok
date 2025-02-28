import requests
import json, math, sys, time
from openai import AzureOpenAI
from dotenv import load_dotenv
import os, time, glob
from tqdm import tqdm
from statistics import mode
from sklearn.metrics import confusion_matrix
import pandas as pd
from tqdm import tqdm

load_dotenv()

def query_o1(prompt):
    client = AzureOpenAI(
      azure_endpoint = os.getenv("o1_endpoint"), 
      api_key=os.getenv("o1_key"),  
      api_version="2024-02-01"
    )
    
    response = client.chat.completions.create(
        model=os.getenv("o1_mini"), # model = "deployment_name".
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content


def annotate_comments(all_ids, comments):
    machine_id = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
    num_machines = int(os.environ.get('SLURM_ARRAY_TASK_COUNT'))
    
    with open("./query_template.txt", "r") as f:
        query_template = f.read()
        
    output_format =  "\n\nResponse format(never deviate, no asteriks or brackets or extra whitespace):\n[Annotation]\n[Explanation]"
    
    # annotations = []
    # reasonings = []
    ids = []
    responses = []
    
    for i in tqdm(range(len(comments))):
        if i%num_machines == machine_id:
            comment = comments[i]
            comment_ids = all_ids[i]
        
            prompt = query_template + comment + output_format
            response = query_o1(prompt)
            
            # annotation, reasoning = response.split("\n")

            ids.append(comment_ids)
            responses.append(response)
            # annotations.append(annotation)
            # reasonings.append(reasoning)
            
            time.sleep(0.01)
    # return ids, annotations, reasonings
    return ids, responses


def main(comments_path, output_folder):
    machine_id = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
    
    df = pd.read_csv(comments_path)
    df = df.sample(n=10000, random_state=43)
    
    all_ids = list(df['id'])
    comments = list(df['text'])

    # all_ids = all_ids[:30]
    # comments = comments[:30]
    
    # time.sleep(machine_id*)
    # ids, annotations, reasonings = annotate_comments(all_ids, comments)
    ids, responses = annotate_comments(all_ids, comments)
        
    gpt_df = pd.DataFrame()
    gpt_df["id"] = ids
    gpt_df["response"] = responses
    # gpt_df["annotation"] = annotations
    # gpt_df["reasoning"] = reasonings
    gpt_df.to_csv(f"{output_folder}/gpt_annotation_mono_machine_{machine_id}.csv", index=None)

if __name__ == "__main__":
    input_path = sys.argv[1]
    output_folder = sys.argv[2]

    main(input_path, output_folder)

