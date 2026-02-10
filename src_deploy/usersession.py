import asyncio
import os
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import logfire
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import Field, SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession


class EmbedUserData(BaseModel):
    user_id: str


class UserSession(SQLModel, table=True):
    id: Optional[int] = Field(primary_key=True)
    user_id: str
    session_id: int
    seconds_used: int
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(ZoneInfo("UTC")).replace(tzinfo=None)
    )


class User(SQLModel, table=True):
    id: str = Field(default=None, primary_key=True)
    name: str
    email: str
    email_verified: bool


_engine: Optional[AsyncEngine] = None


def get_engine() -> AsyncEngine:
    """Lazy initialization of the database engine. Only creates it when first needed."""
    global _engine
    if _engine is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError(
                "DATABASE_URL is not set. Set the environment variable."
            )
        _engine = create_async_engine(database_url, poolclass=NullPool)
    return _engine


async def create_tables() -> None:
    async with get_engine().begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def drop_tables(model: SQLModel) -> None:
    async with get_engine().begin() as conn:
        await conn.run_sync(model.__table__.drop)


async def write_user_session(user_session: UserSession) -> None:
    async with AsyncSession(get_engine()) as session:
        session.add(await session.merge(user_session))
        await session.commit()


async def fetch_user_sessions(user_id: str) -> list[UserSession]:
    async with AsyncSession(get_engine()) as session:
        result = await session.exec(
            select(UserSession)
            .where(UserSession.user_id == user_id)
            .order_by(UserSession.created_at.desc())
        )
        return result.all()


async def fetch_user_info(user_id: str) -> tuple[str, str]:
    async with AsyncSession(get_engine()) as session:
        result = await session.exec(select(User).where(User.id == str(user_id)))
        user = result.first()
        return (user.name, user.email)


async def create_user_session(user_id: str, session_id: int) -> UserSession:
    async with AsyncSession(get_engine()) as session:
        user_session = UserSession(
            user_id=user_id, session_id=session_id, seconds_used=0
        )
        session.add(user_session)
        await session.commit()
        await session.refresh(user_session)
        return user_session


async def user_exists(user_id: str) -> bool:
    async with AsyncSession(get_engine()) as session:
        result = await session.exec(select(User).where(User.id == user_id))
        result = result.first()
        if result is None:
            return False
        elif result.id is None or result.email_verified is False:
            return False
        return True


async def fetch_user_time_used(user_id: str) -> int:
    async with AsyncSession(get_engine()) as session:
        result = await session.exec(
            select(UserSession)
            .where(UserSession.user_id == user_id)
            .order_by(UserSession.created_at.desc())
        )
        return sum(user_session.seconds_used for user_session in result.all())


async def protected_create_session(user_id: str, session_id: int) -> UserSession | None:
    try:
        return await asyncio.wait_for(
            asyncio.shield(create_user_session(user_id, session_id)), timeout=5.0
        )
    except asyncio.TimeoutError:
        logfire.error(
            "The request timed out, but the session is still being created in the background."
        )


async def main():
    # await drop_tables(UserSession)
    # await create_tables()
    print(await fetch_user_time_used("123"))


if __name__ == "__main__":
    asyncio.run(main())
