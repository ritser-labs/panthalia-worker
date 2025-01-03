from sqlalchemy import select, update
from ....models import Perm, PermDescription, PermType, Sot


class DBAdapterPermissionsMixin:
    async def get_perm(self, address: str, perm: int):
        lower_address = address.lower()
        async with self.get_async_session() as session:
            stmt = select(Perm).filter_by(address=lower_address, perm=perm)
            result = await session.execute(stmt)
            perm_obj = result.scalar_one_or_none()
            return perm_obj

    async def set_last_nonce(self, address: str, perm: int, last_nonce: str):
        lower_address = address.lower()
        async with self.get_async_session() as session:
            stmt = (
                update(Perm)
                .where(Perm.address == lower_address, Perm.perm == perm)
                .values(last_nonce=last_nonce)
            )
            await session.execute(stmt)
            await session.commit()

            updated_perm = await session.execute(
                select(Perm).where(Perm.address == lower_address, Perm.perm == perm)
            )
            perm_obj = updated_perm.scalar_one_or_none()
            return perm_obj.id

    async def create_perm(self, address: str, perm: int):
        lower_address = address.lower()
        async with self.get_async_session() as session:
            new_perm = Perm(
                address=lower_address,
                perm=perm
            )
            session.add(new_perm)
            await session.commit()
            await session.refresh(new_perm)
            return new_perm.id

    async def create_perm_description(self, perm_type: PermType):
        async with self.get_async_session() as session:
            new_perm_description = PermDescription(
                perm_type=perm_type
            )
            session.add(new_perm_description)
            await session.commit()
            await session.refresh(new_perm_description)
            return new_perm_description.id

    async def create_sot(self, job_id: int, url: str | None):
        perm_id = await self.create_perm_description(perm_type=PermType.ModifySot)
        async with self.get_async_session() as session:
            new_sot = Sot(
                job_id=job_id,
                perm=perm_id,
                url=url
            )
            session.add(new_sot)
            await session.commit()
            await session.refresh(new_sot)
            return new_sot.id

    async def update_sot(self, sot_id: int, url: str | None):
        async with self.get_async_session() as session:
            stmt = update(Sot).where(Sot.id == sot_id).values(url=url)
            await session.execute(stmt)
            await session.commit()
            return True

    async def get_sot(self, id: int):
        async with self.get_async_session() as session:
            stmt = select(Sot).filter_by(id=id)
            result = await session.execute(stmt)
            sot = result.scalar_one_or_none()
            return sot

    async def get_sot_by_job_id(self, job_id: int):
        async with self.get_async_session() as session:
            stmt = select(Sot).filter_by(job_id=job_id)
            result = await session.execute(stmt)
            sot = result.scalar_one_or_none()
            return sot
