from src.model.bin import Bin

single_bin = [
    Bin(
        enum=1,
        column="value",
        description="Test description 1",
        lower=40,
        upper=60,
        colour="3B9C17",
        opacity=50,
    ),
]
test_bins = [
    Bin(
        enum=1,
        column="value",
        description="Test description 1",
        lower=0,
        upper=20,
        ignore=True,
    ),
    Bin(
        enum=2,
        column="value",
        description="Test description 2",
        lower=20,
        upper=40,
        colour="5CFF21",
        opacity=50,
    ),
    Bin(
        enum=3,
        column="value",
        description="Test description 3",
        lower=40,
        upper=60,
        colour="3B9C17",
        opacity=50,
    ),
    Bin(
        enum=4,
        column="value",
        description="Test description 4",
        lower=60,
        upper=80,
        colour="3B9C17",
        opacity=100,
    ),
    Bin(
        enum=3,
        column="value",
        description="Test description 3",
        lower=80,
        upper=100,
        colour="3B9C17",
        opacity=50,
        boundary_type="[]",
    ),
]
