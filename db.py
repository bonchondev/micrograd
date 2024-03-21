from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column 
from sqlalchemy import create_engine

class Base(DeclarativeBase):...

class BigX(Base):
    __tablename__ = "big_x"
    id: Mapped[int] = mapped_column(primary_key=True)
    first_value: Mapped[float] = mapped_column()
    second_value: Mapped[float] = mapped_column()

    def __repr__(self) -> str:
        return f"X([0]={self.first_value},[1]={self.second_value})"

class SmallY(Base):
    __tablename__ = "small_y"
    id: Mapped[int] = mapped_column(primary_key=True)
    value: Mapped[float] = mapped_column()
    def __repr__(self) -> str:
        return f"y([0]={self.value})"


engine = create_engine("sqlite:///data/dataset.db", echo=True)
