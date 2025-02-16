# spl/master/sot_upload.py
import aiohttp
import os
import time
import boto3
import logging
from botocore.exceptions import BotoCoreError, NoCredentialsError

CHUNK_SIZE = 1024 * 1024 * 5  # 5MB chunks for S3 multi-part

logger = logging.getLogger(__name__)

async def upload_sot_state_if_needed(db_adapter, job_obj):
    """
    If the job had at least 1 Task and is truly 'finished',
    then we stream-download the final SOT state from the SOT, stream-upload it to S3,
    record the upload in the DB, and then prune old usage.
    
    Updated to log errors for missing tasks, missing configuration, and other issues.
    """
    logger.debug(f"Checking if SOT upload is needed for job {job_obj.id}...")
    try:
        # 1) Check if job had at least one task:
        if not await db_adapter.job_has_matched_task(job_obj.id):
            logger.error(f"Job {job_obj.id} has no matched tasks; skipping SOT upload.")
            return

        # 2) Build the SOT /latest_state URL
        sot_url = (job_obj.sot_url or "").rstrip("/")
        if not sot_url:
            logger.error(f"Job {job_obj.id} has no SOT URL set; skipping SOT upload.")
            return
        latest_url = f"{sot_url}/latest_state"

        # 3) Retrieve S3 configuration from environment variables
        s3_bucket = os.getenv("PANTHALIA_S3_BUCKET_NAME")
        s3_folder = os.getenv("PANTHALIA_S3_FOLDER_NAME", "").rstrip("/")
        if not s3_bucket or not s3_folder:
            logger.error("S3 configuration missing: PANTHALIA_S3_BUCKET_NAME or PANTHALIA_S3_FOLDER_NAME not set.")
            return

        user_id = job_obj.user_id
        file_key = f"{s3_folder}/job_{job_obj.id}_final_{int(time.time())}.safetensors"
        logger.info(f"Starting SOT state upload for job {job_obj.id} to S3 key: {file_key}")

        # 4) Create S3 client
        session = boto3.session.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        s3_client = session.client('s3', region_name='us-east-1')  # adjust region as needed

        # Initiate a multipart upload
        create_resp = s3_client.create_multipart_upload(Bucket=s3_bucket, Key=file_key)
        upload_id = create_resp['UploadId']
        parts = []
        part_number = 1
        total_bytes = 0

        try:
            async with aiohttp.ClientSession() as http_sess:
                async with http_sess.get(latest_url) as r:
                    if r.status != 200:
                        logger.error(f"SOT endpoint {latest_url} returned HTTP {r.status}")
                        raise RuntimeError(f"SOT returned HTTP {r.status} for {latest_url}")

                    # Read in chunks and upload parts as they accumulate.
                    chunk_buffer = bytearray()
                    while True:
                        chunk = await r.content.read(CHUNK_SIZE)
                        if not chunk:
                            # If there is remaining data, upload it as the final part.
                            if chunk_buffer:
                                etag = await _upload_part(s3_client, s3_bucket, file_key, upload_id, part_number, chunk_buffer)
                                parts.append({"ETag": etag, "PartNumber": part_number})
                                total_bytes += len(chunk_buffer)
                                chunk_buffer.clear()
                            break
                        chunk_buffer.extend(chunk)
                        if len(chunk_buffer) >= CHUNK_SIZE:
                            etag = await _upload_part(s3_client, s3_bucket, file_key, upload_id, part_number, chunk_buffer)
                            parts.append({"ETag": etag, "PartNumber": part_number})
                            total_bytes += len(chunk_buffer)
                            part_number += 1
                            chunk_buffer.clear()
        except Exception as e:
            logger.error(f"Error during streaming download/upload from SOT: {e}", exc_info=True)
            s3_client.abort_multipart_upload(Bucket=s3_bucket, Key=file_key, UploadId=upload_id)
            return

        if total_bytes == 0:
            logger.error("No data received from SOT; aborting multipart upload.")
            s3_client.abort_multipart_upload(Bucket=s3_bucket, Key=file_key, UploadId=upload_id)
            return

        try:
            s3_client.complete_multipart_upload(
                Bucket=s3_bucket,
                Key=file_key,
                MultipartUpload={"Parts": parts},
                UploadId=upload_id
            )
            logger.info(f"Completed S3 multipart upload for job {job_obj.id}: {total_bytes} bytes uploaded.")
        except Exception as e:
            logger.error(f"Error completing S3 multipart upload: {e}", exc_info=True)
            s3_client.abort_multipart_upload(Bucket=s3_bucket, Key=file_key, UploadId=upload_id)
            return

        # 6) Record the SOT upload in the DB.
        try:
            sot_upload_id = await db_adapter.record_sot_upload(
                job_id=job_obj.id,
                user_id=user_id,
                s3_key=file_key,
                file_size_bytes=total_bytes
            )
            logger.info(f"Recorded SOT upload in DB with id {sot_upload_id} for job {job_obj.id}.")
        except Exception as e:
            logger.error(f"Error recording SOT upload in DB: {e}", exc_info=True)
            return

        # 7) Prune old SOT uploads if needed.
        try:
            await db_adapter.prune_old_sot_uploads(user_id)
            logger.info(f"Pruned old SOT uploads for user {user_id}.")
        except Exception as e:
            logger.error(f"Error pruning old SOT uploads: {e}", exc_info=True)

    except Exception as outer_e:
        logger.error(f"Unhandled exception in upload_sot_state_if_needed: {outer_e}", exc_info=True)
        raise

async def _upload_part(s3_client, bucket, key, upload_id, part_number, chunk_buffer):
    """
    Helper that performs the S3 upload_part for a given chunk.
    """
    resp = s3_client.upload_part(
        Bucket=bucket,
        Key=key,
        PartNumber=part_number,
        UploadId=upload_id,
        Body=bytes(chunk_buffer)
    )
    return resp['ETag']
