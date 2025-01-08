from sqlalchemy import select, update
from ....models import Perm, PermDescription, PermType, Sot
from typing import Optional

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

    async def create_perm_description(self, perm_type: PermType, restricted_sot_id: int | None):
        async with self.get_async_session() as session:
            new_perm_description = PermDescription(
                perm_type=perm_type,
                restricted_sot_id=restricted_sot_id
            )
            session.add(new_perm_description)
            await session.commit()
            await session.refresh(new_perm_description)
            return new_perm_description.id
    
    async def get_perm_description(self, perm_desc_id: int):
        async with self.get_async_session() as session:
            stmt = select(PermDescription).filter_by(id=perm_desc_id)
            result = await session.execute(stmt)
            perm_desc = result.scalar_one_or_none()
            return perm_desc


    async def create_sot(self, job_id: int, address: str | None, url: str | None) -> Optional[int]:
        """
        Creates a SoT (State of Truth) with perm_type=ModifySot. This ensures
        the server_auth logic sees 'ModifySot' and won't fail with 403 'Not a ModifySot perm desc'.
        """
        async with self.get_async_session() as session:
            # 1) Create a perm_description row with perm_type=ModifySot
            perm_id = await self.create_perm_description(
                perm_type=PermType.ModifySot,         # <--- IMPORTANT: Must be ModifySot
                restricted_sot_id=None
            )

            # 2) Create the SoT row referencing that new perm_description
            new_sot = Sot(
                job_id=job_id,
                perm=perm_id,    # SoT row points to the ModifySot permission we just made
                url=url
            )
            session.add(new_sot)
            await session.flush()   # flush so new_sot.id is assigned

            # 3) Update the perm_description row to tie it back to this SoT's id
            pd_stmt = (
                update(PermDescription)
                .where(PermDescription.id == perm_id)
                .values(restricted_sot_id=new_sot.id)
            )
            await session.execute(pd_stmt)

            # 4) Commit and return the new_sot.id
            await session.commit()
            await session.refresh(new_sot)
            await self.create_perm(address, perm_id)
            return new_sot.id


    async def update_sot(self, sot_id: int, url: str | None):
        async with self.get_async_session() as session:
            stmt = update(Sot).where(Sot.id == sot_id).values(url=url)
            await session.execute(stmt)
            await session.commit()
            return True

    async def get_sot(self, sot_id: int):
        async with self.get_async_session() as session:
            stmt = select(Sot).filter_by(id=sot_id)
            result = await session.execute(stmt)
            sot = result.scalar_one_or_none()
            return sot

    async def get_sot_by_job_id(self, job_id: int):
        async with self.get_async_session() as session:
            stmt = select(Sot).filter_by(job_id=job_id)
            result = await session.execute(stmt)
            sot = result.scalar_one_or_none()
            return sot
