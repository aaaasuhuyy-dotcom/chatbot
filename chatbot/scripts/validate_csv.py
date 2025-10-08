import csv
from pathlib import Path

CSV_PATH = Path(r"d:\chat_ai\dataset\data_mentah.csv")


def main():
    with CSV_PATH.open('r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        bad = []
        total = 0
        for i, row in enumerate(reader, start=1):
            total += 1
            if len(row) != 4:
                bad.append((i, len(row), row[:3]))
        print(f"Total rows: {total}")
        print(f"Bad rows: {len(bad)}")
        for item in bad[:20]:
            print(item)


if __name__ == '__main__':
    main()
