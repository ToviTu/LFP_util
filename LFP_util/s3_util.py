from pickle import TRUE
import numpy as np
import neuraltoolkit as ntk
import glob
import csv
import os
import smart_open
import boto3
import tqdm
import argparse
import fnmatch
import io
import re
import tempfile
from pathlib import Path


def get_remote_files(prefix: str):
    try:
        response = client.list_objects_v2(Bucket="hengenlab", Prefix=prefix)["Contents"]
    except:
        response = []
    return ntk.ntk_videos.natural_sort([file["Key"] for file in response])


def get_s3_client():
    with open(f"{Path.home()}/.aws/credentials", "r") as f:
        for line in f:
            if "aws_access_key_id" in line:
                id = re.search(r"\W(\w+)", line).group(1)
            if "aws_secret_access_key" in line:
                key = re.search(r"\W(\w+)", line).group(1)
        return boto3.Session(aws_access_key_id=id, aws_secret_access_key=key).client(
            "s3", endpoint_url="https://s3-central.nrp-nautilus.io"
        )
