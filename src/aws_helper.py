import io
import os

import boto3
import botocore
import cv2
import numpy as np
import pickle
from PIL import Image


S3 = boto3.resource('s3', region_name='us-west-2')
S3_client = boto3.client('s3')


def upload_np_to_s3(bucket: str, key: str, image: np.array) -> None:
    """
    :param bucket: bucket name
    :param key: keypath coco_train_np_imgs/COCOtrain_31987491384.np
    :param image: normalized resized image as np.array
    :return: None
    """
    image_data = io.BytesIO()
    pickle.dump(image, image_data)
    image_data.seek(0)
    S3_client.upload_fileobj(image_data, bucket, key)


def upload_img_to_s3(bucket: str, key: str, image: np.array):
    """
    :bucket: the bucket to upload the image to
    :key: s3 partition key
    :image: np representation of the image
    :tagging: Tags associated to image
    converts numpy image to a cv2 image, encodes as jpg and uploads to s3
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
    S3_client.put_object(Bucket=bucket, Key=key, Body=image_bytes)


def read_image(bucket: str, key: str):
    """
    Reads an image from the given s3 bucket.
    :param bucket: the s3 bucket
    :param key: the object key to download
    :return: an cv2 image
    """
    filename = f'/tmp/{key.split("/")[-1]}'
    try:
        S3.Bucket(bucket).download_file(key, filename)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise e
    try:
        image = Image.open(filename)
        print(f"format: {image.format},size: {image.size}, mode:{image.mode}")
        np_image = np.array(image)
    except Exception as e:
        print(f"Cannot read {filename} from /tmp")
        raise e
    os.remove(filename)
    return np_image

