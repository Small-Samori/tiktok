import pandas as pd
import os, sys

def remove_short_text(df, col, threshold=5):
    
    df['split'] = [i.split(' ') for i in df[col]]
    lengths = [len(i) for i in df['split']]
    
    drop_ids = [i for i in range(len(lengths)) if lengths[i] < threshold]
    df = df.drop(drop_ids)
    df = df.drop(['split', 'index'], axis=1)
    df = df.reset_index().drop(['index'], axis=1)
    
    return df

def clean_comments(df, col="text",threshold=5):
    df = df.dropna(subset=[col])
    df = df.drop_duplicates(subset=["id"])
    df = df.drop_duplicates(subset=[col]).reset_index()
    df = remove_short_text(df, col=col, threshold=threshold)
    
    return df

def main(input_path, output_path):
    comments_df = pd.read_csv(input_path)
    
    cleaned_comments_df = clean_comments(comments_df)
    cleaned_comments_df.to_csv(output_path, index=None)

if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    main(input_path, output_path)