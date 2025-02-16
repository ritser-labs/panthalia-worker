# spl/master/jobs.py

import aiohttp
import os
import time
import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError

CHUNK_SIZE = 1024 * 1024 * 5  # 5MB chunks for S3 multi-part

async def upload_sot_state_if_needed(db_adapter, job_obj):
    """
    If the job had at least 1 Task and is truly 'finished',
    then we stream-download final SOT state from SOT => to S3 => record DB => prune old usage.
    """
    # 1) Check if job had at least one task:
    task_count = await db_adapter.get_task_count_for_job(job_obj.id)
    if task_count < 1:
        return  # do nothing, user has no tasks

    # 2) Build the SOT /latest_state url
    #    The job_obj.sot_url might look like 'http://1.2.3.4:5001'
    sot_url = (job_obj.sot_url or "").rstrip("/")
    if not sot_url:
        return
    latest_url = f"{sot_url}/latest_state"
    
    # 3) We'll create an S3 client with env vars
    #    Make sure you have these env variables:
    #      AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
    #      PANTHALIA_S3_BUCKET_NAME, PANTHALIA_S3_FOLDER_NAME
    s3_bucket = os.getenv("PANTHALIA_S3_BUCKET_NAME")
    s3_folder = os.getenv("PANTHALIA_S3_FOLDER_NAME", "").rstrip("/")
    if not s3_bucket or not s3_folder:
        return  # not configured

    user_id = job_obj.user_id
    file_key = f"{s3_folder}/job_{job_obj.id}_final_{int(time.time())}.safetensors"

    # 4) Attempt streaming download from SOT => streaming upload to S3
    session = boto3.session.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    s3_client = session.client('s3', region_name='us-east-1')  # or your region

    # We'll do a "multipart upload" so we don't have to store anything on disk
    create_resp = s3_client.create_multipart_upload(Bucket=s3_bucket, Key=file_key)
    upload_id = create_resp['UploadId']
    parts = []
    part_number = 1
    total_bytes = 0

    try:
        async with aiohttp.ClientSession() as http_sess:
            async with http_sess.get(latest_url) as r:
                if r.status != 200:
                    # If the SOT doesn't have final or is small, we skip
                    raise RuntimeError(f"SOT returned HTTP {r.status} for {latest_url}")
                
                # We read in chunk_size increments
                chunk_buffer = bytearray()
                while True:
                    chunk = await r.content.read(CHUNK_SIZE)
                    if not chunk:
                        # no more data
                        if chunk_buffer:
                            # finalize the last part
                            etag = await _upload_part(
                                s3_client, s3_bucket, file_key, upload_id,
                                part_number, chunk_buffer
                            )
                            parts.append({"ETag": etag, "PartNumber": part_number})
                            part_number += 1
                            total_bytes += len(chunk_buffer)
                            chunk_buffer.clear()
                        break
                    
                    # If we got a chunk, store it in chunk_buffer
                    chunk_buffer.extend(chunk)
                    # If it hits CHUNK_SIZE, upload as a new part
                    if len(chunk_buffer) >= CHUNK_SIZE:
                        etag = await _upload_part(
                            s3_client, s3_bucket, file_key, upload_id,
                            part_number, chunk_buffer
                        )
                        parts.append({"ETag": etag, "PartNumber": part_number})
                        part_number += 1
                        total_bytes += len(chunk_buffer)
                        chunk_buffer.clear()

        # If we never got data => total_bytes = 0 => skip?
        if total_bytes == 0:
            # Possibly abort the multipart
            s3_client.abort_multipart_upload(Bucket=s3_bucket, Key=file_key, UploadId=upload_id)
            return

        # 5) Complete multipart
        s3_client.complete_multipart_upload(
            Bucket=s3_bucket,
            Key=file_key,
            MultipartUpload={"Parts": parts},
            UploadId=upload_id
        )

    except Exception as exc:
        # try to clean up partial upload
        s3_client.abort_multipart_upload(Bucket=s3_bucket, Key=file_key, UploadId=upload_id)
        raise

    # 6) We have a final S3 object of total_bytes. Record in DB
    sot_upload_id = await db_adapter.record_sot_upload(
        job_id=job_obj.id,
        user_id=user_id,
        s3_key=file_key,
        file_size_bytes=total_bytes
    )

    # 7) Now prune usage if needed
    await db_adapter.prune_old_sot_uploads(user_id)

async def _upload_part(s3_client, bucket, key, upload_id, part_number, chunk_buffer):
    """
    Helper that does the actual S3 upload_part for one chunk.
    """
    resp = s3_client.upload_part(
        Bucket=bucket,
        Key=key,
        PartNumber=part_number,
        UploadId=upload_id,
        Body=bytes(chunk_buffer)
    )
    return resp['ETag']
