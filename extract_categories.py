import csv
import os
import sys
import duckdb

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

output_dir  = "output_draft"
kalshi = "data/kalshi/markets/*.parquet"

os.makedirs(output_dir, exist_ok=True)


def discover_prefixes():
    con = duckdb.connect()

    rows = con.execute(f"""
        SELECT
            SPLIT_PART(event_ticker, '-', 1) AS prefix,
            COUNT(*) AS total_markets,
            SUM(CASE WHEN result IN ('yes','no') THEN 1 ELSE 0 END) AS resolved,
            SUM(CASE WHEN result IN ('yes','no')
                      AND volume >= 100 THEN 1 ELSE 0 END) AS usable,
            ROUND(AVG(
                ARRAY_LENGTH(STR_SPLIT(TRIM(title), ' '))
            ), 1) AS avg_word_count,
            ANY_VALUE(title) AS sample_title
        FROM '{kalshi}'
        WHERE event_ticker IS NOT NULL
        GROUP BY prefix
        ORDER BY usable DESC
    """).fetchall()

    con.close()
    
    return rows


def save_csv(rows):
    path = f"{output_dir}/kalshi_all_prefixes.csv"
    header = ["prefix", "total_markets", "resolved", "usable",
              "avg_word_count", "sample_title"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return path


if __name__ == "__main__":
    print("Scan all Kalshi parquet files for unique event_ticker PREFIX")
    rows = discover_prefixes()

    for prefix, total, resolved, usable, avg_w, sample in rows[:10]:
        print(f"  {prefix:<35} {usable:>8,}   {(sample or '')[:55]}")

    save_csv(rows)

#build_domain next
