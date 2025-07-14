from lib.interact.tile import Tile


def print_map(grid: list[list["Tile | None"]], print_range: range) -> None:
    assert grid
    assert len(grid) >= len(print_range)

    print("\t.", end="")
    for i in print_range:
        print(f" {i} ", end=", ")

    print("")
    for i, row in enumerate(grid[print_range[0] : print_range[-1]]):
        assert len(row) >= len(print_range)
        print(f"{i + 75}", end="\t")
        print(
            [col.tile_type.ljust(2, " ") if col else "__" for col in row[75:96]],
            flush=True,
        )

    print(
        "",
        end="### END ###\n",
    )
