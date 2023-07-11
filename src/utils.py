import boto3
from boto3 import client
import os
import pandas as pd

def filter_headlines(target, df, text_field):
    """
    Placeholder Docstring
    Adds a field(bool) if the string in target is found in the string in text_field
    """
    listed = df[text_field].tolist()
    df["on_brand"] = [target in n for n in listed]
    df_on_brand = df[df['on_brand'] == True]

    df.drop(df[df['on_brand'] >= True].index)
    listed = df[text_field].tolist()
    competitor_str = target.split()[1]
    df["competitors"] = [competitor_str in n for n in listed]
    df_competitors = df[df['competitors'] == True]

    return df_on_brand, df_competitors

def check_dir(path, create=True):
    """
    Placeholder Docstring.
    Checks if path exists and makes a folder if it doesn't
    """        
    if os.path.exists(path):
        return True
    elif create:
        os.makedirs(path)
    else:
        return False


def clean_dataframe(df, fields):
    """
    Placeholder Docstring.
    Some dataframe cleaning steps. Incomplete. 
    """      
    df = df.dropna(subset=fields)
    filtered_df = df[df['language'] == "en"]
    return filtered_df

class S3Connector():
    """
    Placeholder Docstring
    Allows listing files in a bucket and downloading them. 
    """
    def __init__(self):
        self.s3_conn = boto3.resource('s3',
            aws_access_key_id=os.environ["aws_access_key_id"],
            aws_secret_access_key=os.environ["aws_secret_access_key"]
            )
        
    def list_files(self, bucket_name, s3_folder):
        """
        List the contents of a folder directory
        Params:
            bucket_name: the name of the s3 bucket
            s3_folder: the folder path in the s3 bucket
        """
        bucket = self.s3_conn.Bucket(bucket_name)
        file_list = []
        for obj in bucket.objects.filter():
            file_list.append(os.path.relpath(obj.key, s3_folder))
        return file_list

    def download_s3_folder(self, bucket_name, s3_folder, local_dir=None):
        """
        Download the contents of a folder directory
        Params:
            bucket_name: the name of the s3 bucket
            s3_folder: the folder path in the s3 bucket
            local_dir: a relative or absolute directory path in the local file system
        """
        bucket = self.s3_conn.Bucket(bucket_name)
        for obj in bucket.objects.filter(Prefix=s3_folder):
            target = obj.key if local_dir is None \
                else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
            if not os.path.exists(os.path.dirname(target)):
                os.makedirs(os.path.dirname(target))
            if obj.key[-1] == '/':
                continue
            bucket.download_file(obj.key, target)
