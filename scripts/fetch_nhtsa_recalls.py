"""Pull real recall records from the public NHTSA API and convert them to this project's schema.

Example:
    python scripts/fetch_nhtsa_recalls.py --make toyota --model "prius" --years 2022 2023 \
        --out data/recalls/nhtsa_prius.json
Then point RECALLS_PATH at the output file.
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from datetime import datetime

API = "https://api.nhtsa.gov/recalls/recallsByVehicle"


def fetch(make: str, model: str, year: int) -> list[dict]:
    query = urllib.parse.urlencode({"make": make, "model": model, "modelYear": year})
    with urllib.request.urlopen(f"{API}?{query}", timeout=30) as resp:
        return json.load(resp).get("results", [])


def parse_date(value: str) -> str:
    for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except (ValueError, TypeError):
            continue
    return value or ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--make", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records: dict[str, dict] = {}
    for year in args.years:
        for item in fetch(args.make, args.model, year):
            cid = item.get("NHTSACampaignNumber", "")
            rec = records.setdefault(
                cid,
                {
                    "campaign_id": cid,
                    "model": args.model.title(),
                    "years": [],
                    "component": item.get("Component", ""),
                    "summary": item.get("Summary", ""),
                    "remedy": item.get("Remedy", ""),
                    "status": "open",
                    "date": parse_date(item.get("ReportReceivedDate", "")),
                },
            )
            if year not in rec["years"]:
                rec["years"].append(year)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(list(records.values()), f, indent=2)
    print(f"Wrote {len(records)} recalls to {args.out}")


if __name__ == "__main__":
    main()
