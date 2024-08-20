from pydantic import BaseModel
from datetime import date


class User(BaseModel):
    name: str = None
    id: int
    age: int


class Item(BaseModel):
    id: int
    genre: str
    year: date
    actor: list[str]
