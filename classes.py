from dataclasses import dataclass


@dataclass
class Bin:
    column: str
    enum: int
    description: str
    lower_bound: int
    upper_bound: int
    colour: str = None
