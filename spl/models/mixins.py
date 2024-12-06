from datetime import datetime
from pytz import UTC
from sqlalchemy import Column, DateTime
import enum

class Serializable:
    __abstract__ = True
    def as_dict(self):
        result = {}
        for c in self.__table__.columns:
            value = getattr(self, c.name)
            # If value is enum, extract value.name if it's an Enum instance
            result[c.name] = value.name if isinstance(value, enum.Enum) else value
        return result

class TimestampMixin:
    last_updated = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))
    submitted_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))
