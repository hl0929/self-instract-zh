import jsonlines


def read_jsonline(path: str) -> list[dict]:
    with jsonlines.open(path) as f:
        data = list(f)
    return data


def save_jsonline(path: str, data) -> None:
    with jsonlines.open(path, "w") as f:
        for line in data:
            f.write(line)
