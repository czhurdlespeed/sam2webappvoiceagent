import asyncio
import os
from datetime import datetime
from typing import List, Optional
from zoneinfo import ZoneInfo

from pydantic import BaseModel, EmailStr, NonNegativeInt, PositiveInt
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import Field, Relationship, SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession


class EmbedUserData(BaseModel):
    user_id: str


class UserAgentUsage(BaseModel):
    user_id: str
    name: str
    email: EmailStr
    email_verified: bool
    session_ids: list[PositiveInt]
    seconds_used: list[NonNegativeInt]


class UserSession(SQLModel, table=True):
    id: Optional[int] = Field(primary_key=True)
    user_id: str = Field(foreign_key="user.id", ondelete="CASCADE")
    session_id: int = Field(nullable=False, default=None, gt=0)
    seconds_used: int
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(ZoneInfo("UTC")).replace(tzinfo=None)
    )

    user: "User" = Relationship(back_populates="sessions")


class User(SQLModel, table=True):
    id: str = Field(default=None, primary_key=True)
    name: str
    email: str
    email_verified: bool

    sessions: list[UserSession] = Relationship(
        back_populates="user", cascade_delete=True
    )


class VoiceAgentUsageDatabase:
    async def __aenter__(self):
        self.engine = create_async_engine(str(os.getenv("DATABASE_URL")))
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.engine.dispose()

    async def create_tables(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

    async def drop_tables(self, model: SQLModel) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(model.__table__.drop)

    async def fetch_user_time_used(self, user_id: str) -> int:
        async with AsyncSession(self.engine) as session:
            result = await session.exec(
                select(UserSession)
                .where(UserSession.user_id == user_id)
                .order_by(UserSession.created_at.desc())
            )
            return sum(user_session.seconds_used for user_session in result.all())

    async def write_user_session(self, user_session: UserSession) -> UserSession:
        async with AsyncSession(self.engine) as session:
            session.add(user_session)
            await session.commit()
            await session.refresh(user_session)
            return user_session

    async def user_exists(self, user_id: str) -> bool:
        async with AsyncSession(self.engine) as session:
            result = await session.exec(select(User).where(User.id == user_id))
            result = result.first()
            if result is None:
                return False
            elif result.id is None or result.email_verified is False:
                return False
            return True

    async def fetch_user_info_and_sessions(self, user_id: str) -> UserAgentUsage | None:
        async with AsyncSession(self.engine) as session:
            result = await session.exec(
                select(  # type: ignore[call-overload]
                    User.id,
                    User.name,
                    User.email,
                    User.email_verified,
                    UserSession.session_id,
                    UserSession.seconds_used,
                )
                .where(User.id == user_id)
                .join(UserSession, isouter=True)
            )
            rows = result.all()
            if not rows:
                return None
            return UserAgentUsage(
                user_id=user_id,
                name=rows[0].name,
                email=rows[0].email,
                email_verified=rows[0].email_verified,
                session_ids=[
                    row.session_id for row in rows if row.session_id is not None
                ],
                seconds_used=[
                    row.seconds_used for row in rows if row.seconds_used is not None
                ],
            )


async def main():
    async with VoiceAgentUsageDatabase() as database:
        user_agent_usage: List[
            UserAgentUsage
        ] = await database.fetch_user_info_and_sessions(
            "GUgn2e2x6v74nN4RmeXzZC7NZFOT2Crj"
        )
    print(user_agent_usage)


if __name__ == "__main__":
    asyncio.run(main())
