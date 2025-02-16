# file: spl/db/server/adapter/sot_uploads.py
import time
import boto3
from datetime import datetime, timedelta
from sqlalchemy import select, func, asc
from ....models import SotUpload
from ....models import Job

MAX_PER_USER_BYTES = 1_000_000_000_000  # 1 TB

class DBAdapterSotUploadsMixin:
    async def record_sot_upload(self, job_id: int, user_id: str, s3_key: str, file_size_bytes: int):
        async with self.get_async_session() as session:
            # confirm job is real & belongs to user
            job_obj = await session.get(Job, job_id)
            if not job_obj:
                return None
            if job_obj.user_id != user_id:
                # or if you allow an admin to do it on behalf, that's up to you
                raise ValueError("Job does not belong to user.")
            
            new_rec = SotUpload(
                job_id=job_id,
                user_id=user_id,
                s3_key=s3_key,
                file_size_bytes=file_size_bytes
            )
            session.add(new_rec)
            await session.commit()
            await session.refresh(new_rec)
            return new_rec.id

    async def get_sot_upload_usage(self, user_id: str) -> int:
        """
        Sum up file_size_bytes for all SotUploads for the user. 
        """
        async with self.get_async_session() as session:
            stmt = select(func.sum(SotUpload.file_size_bytes)).where(SotUpload.user_id == user_id)
            res = await session.execute(stmt)
            total = res.scalar() or 0
            return total

    async def prune_old_sot_uploads(self, user_id: str) -> bool:
        """
        1) If user's total usage is above 1TB, delete oldest records first, 
           continuing until total usage is within the limit or no records left.
        2) Also remove any older than 48 hours from DB. 
           (S3 is removed automatically by lifecycle, but we keep the DB in sync.)
        """
        async with self.get_async_session() as session:
            # Step A: remove 48h old records from DB
            cutoff_time = datetime.utcnow() - timedelta(hours=48)
            old_stmt = select(SotUpload).where(
                SotUpload.created_at < cutoff_time,
                SotUpload.user_id == user_id
            ).order_by(SotUpload.id.asc())
            old_res = await session.execute(old_stmt)
            old_list = old_res.scalars().all()
            for rec in old_list:
                session.delete(rec)
            await session.flush()

            # Step B: if usage > 1TB => remove oldest until below threshold
            # first, do an updated usage check
            stmt_usage = select(func.sum(SotUpload.file_size_bytes)).where(SotUpload.user_id == user_id)
            res_usage = await session.execute(stmt_usage)
            total_usage = res_usage.scalar() or 0

            if total_usage > MAX_PER_USER_BYTES:
                # remove oldest first
                fetch_stmt = (
                    select(SotUpload)
                    .where(SotUpload.user_id == user_id)
                    .order_by(SotUpload.created_at.asc())
                )
                rows_res = await session.execute(fetch_stmt)
                rows = rows_res.scalars().all()
                for r in rows:
                    session.delete(r)
                    total_usage -= r.file_size_bytes
                    if total_usage <= MAX_PER_USER_BYTES:
                        break
            await session.commit()
        return True
