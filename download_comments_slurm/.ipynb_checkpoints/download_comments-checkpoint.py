from utils import *
import requests
import dotenv
import os, time
import pandas as pd
import glob, json

def main():
    oak = os.getenv('OAK')
    video_folder = f"{oak}/samori/tiktok/videos"
    comments_folder = f"{oak}/samori/tiktok/comments"
    
    video_data = pd.read_csv(f"{video_folder}/combined/all_months.csv")
    video_ids = list(video_data['id'])
    
    
    fields = "id,video_id,text,like_count,reply_count,parent_comment_id,create_time"
    cursor = 0
    save_folder = f"{comments_folder}/downloads"
    
    get_comments(video_ids, fields, cursor, save_folder)

if __name__ == "__main__":
    main()