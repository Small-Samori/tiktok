import os, glob
import pandas as pd
from pandas.errors import EmptyDataError
from tqdm import tqdm

oak = os.getenv('OAK')
video_folder = f"{oak}/samori/tiktok/videos"
comments_folder = f"{oak}/samori/tiktok/comments"

comment_csvs = glob.glob(f"{comments_folder}/downloads/*.csv")
comments_combined = pd.read_csv(comment_csvs[0])
for path in tqdm(comment_csvs[1:]):
    df = pd.read_csv(path)
    comments_combined = pd.concat([comments_combined, df], ignore_index=True)
    
comments_combined = comments_combined.drop_duplicates(subset='id', keep='first')
comments_combined.to_csv(f"{comments_folder}/combined_comments_6.csv", index=False)