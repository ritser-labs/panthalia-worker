from sqlalchemy import select, func
from ....models import StateUpdate


class DBAdapterStateUpdatesMixin:
    async def create_state_update(self, job_id: int, data: dict):
        async with self.get_async_session() as session:
            new_state_update = StateUpdate(
                job_id=job_id,
                data=data
            )
            session.add(new_state_update)
            await session.commit()
            await session.refresh(new_state_update)
            return new_state_update.id

    async def get_total_state_updates_for_job(self, job_id: int):
        async with self.get_async_session() as session:
            stmt = select(func.count(StateUpdate.id)).filter_by(job_id=job_id)
            result = await session.execute(stmt)
            total_state_updates = result.scalar_one()
            return total_state_updates
