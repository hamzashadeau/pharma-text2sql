"""Generate the synthetic pharmaceutical sales dataset.

Entirely fabricated: molecules, corporations and products are invented names,
the figures come from a seeded random generator. No real market data, no real
company data — the point is to exercise the pipeline, not to publish numbers.

    python data/generate.py              # ~120k rows -> data/sales.parquet
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import duckdb
import pandas as pd

SEED = 20260101
OUT = Path(__file__).resolve().parent / "sales.parquet"

PERIODS = [f"{y}-{m:02d}" for y in range(2021, 2026) for m in range(1, 13)]

# ATC level 1 is the real anatomical classification; everything below it is made up.
ATC = {
    "A — Alimentary tract and metabolism": [
        ("A02 — Acid related disorders", "A02B — Peptic ulcer", "A02BC — Proton pump inhibitors"),
        ("A10 — Drugs used in diabetes", "A10B — Blood glucose lowering", "A10BA — Biguanides"),
    ],
    "C — Cardiovascular system": [
        ("C07 — Beta blocking agents", "C07A — Beta blocking agents", "C07AB — Selective"),
        ("C09 — Renin-angiotensin", "C09A — ACE inhibitors", "C09AA — ACE inhibitors, plain"),
        ("C10 — Lipid modifying agents", "C10A — Lipid modifying, plain", "C10AA — Statins"),
    ],
    "J — Antiinfectives for systemic use": [
        ("J01 — Antibacterials", "J01C — Beta-lactam antibacterials", "J01CA — Penicillins"),
        ("J05 — Antivirals", "J05A — Direct acting antivirals", "J05AB — Nucleosides"),
    ],
    "N — Nervous system": [
        ("N02 — Analgesics", "N02A — Opioids", "N02AA — Natural opium alkaloids"),
        ("N05 — Psycholeptics", "N05B — Anxiolytics", "N05BA — Benzodiazepines"),
        ("N06 — Psychoanaleptics", "N06A — Antidepressants", "N06AB — SSRIs"),
    ],
    "L — Antineoplastic and immunomodulating agents": [
        ("L01 — Antineoplastic agents", "L01X — Other antineoplastics", "L01XE — Kinase inhibitors"),
        ("L04 — Immunosuppressants", "L04A — Immunosuppressants", "L04AB — TNF alpha inhibitors"),
    ],
}

MOLECULES = {
    "A02BC": ["OMEZARIN", "PANTOVEX", "LANSOTIDE"],
    "A10BA": ["METFORVIN", "GLUCOFEN"],
    "C07AB": ["BISOPRANOL", "METOCARDIN"],
    "C09AA": ["RAMIPRINE", "ENALAVIR", "PERINDOLAN"],
    "C10AA": ["ATORVASTIN", "ROSUVALEN", "SIMVACOR"],
    "J01CA": ["AMOXIVAN", "PIPERACIN"],
    "J05AB": ["ACICLOVEX", "VALGANCIR"],
    "N02AA": ["MORPHINEX", "CODELIRA"],
    "N05BA": ["DIAZEPINOL", "LORAZEVAN"],
    "N06AB": ["SERTRALEX", "FLUOXAMINE", "ESCITALOR"],
    "L01XE": ["IMATINOVIR", "ERLOTINEX", "NILOTIVAR"],
    "L04AB": ["ADALIMAB", "ETANERVAX"],
}

CORPORATIONS = [
    "Aldebaran Pharma", "Borealis Therapeutics", "Cassiopea Health",
    "Draconis Laboratories", "Eridanus Biotech", "Fornax Pharmaceuticals",
    "Gemini Medical", "Hydra Life Sciences",
]

COUNTRIES = [
    ("France", "Europe", "Western Europe"),
    ("Germany", "Europe", "Western Europe"),
    ("Spain", "Europe", "Southern Europe"),
    ("Italy", "Europe", "Southern Europe"),
    ("United Kingdom", "Europe", "Western Europe"),
    ("Poland", "Europe", "Eastern Europe"),
    ("United States", "Americas", "North America"),
    ("Brazil", "Americas", "Latin America"),
    ("Japan", "Asia Pacific", "East Asia"),
    ("Australia", "Asia Pacific", "Oceania"),
    ("Morocco", "Africa", "North Africa"),
    ("South Africa", "Africa", "Southern Africa"),
]

FORMS = ["Tablet", "Capsule", "Oral solution", "Injection", "Patch"]
STRENGTHS = ["5 MG", "10 MG", "20 MG", "40 MG", "80 MG", "100 MG", "250 MG", "500 MG"]
PACK_SIZES = [7, 14, 28, 30, 60, 90]
CHANNELS = ["Retail", "Hospital"]


def build() -> pd.DataFrame:
    rng = random.Random(SEED)
    rows: list[dict] = []

    # A stable identity per (molecule, corporation) pair, so a product keeps the
    # same manufacturer and form across every country and quarter.
    products: list[dict] = []
    for atc1, levels in ATC.items():
        for atc2, atc3, atc4 in levels:
            code = atc4.split(" ")[0]
            for molecule in MOLECULES[code]:
                corp = rng.choice(CORPORATIONS)
                form = rng.choice(FORMS)
                # Two to four marketed presentations per molecule: a strength and
                # a pack size, the way a real product master would hold them.
                for strength in rng.sample(STRENGTHS, rng.randint(2, 4)):
                    pack_size = rng.choice(PACK_SIZES)
                    products.append({
                        "atc1": atc1, "atc2": atc2, "atc3": atc3, "atc4": atc4,
                        "molecule": molecule,
                        "corporation": corp,
                        "manufacturer": f"{corp.split()[0]} Manufacturing",
                        "product": f"{molecule.capitalize()} {strength}",
                        "form": form,
                        "strength": strength,
                        "pack_size": pack_size,
                        # base demand and monthly trend, drawn once per presentation
                        "_base": rng.uniform(120, 14_000),
                        "_trend": rng.uniform(-0.006, 0.013),
                        "_price": rng.uniform(1.4, 240.0),
                    })

    for country, region, sub_region in COUNTRIES:
        country_weight = rng.uniform(0.25, 2.4)
        for p in products:
            # Not every presentation is marketed in every country.
            if rng.random() < 0.35:
                continue
            channel = rng.choice(CHANNELS) if rng.random() < 0.3 else None
            channels = [channel] if channel else CHANNELS
            for channel in channels:
                channel_weight = 1.0 if channel == "Retail" else rng.uniform(0.2, 0.6)
                for i, period in enumerate(PERIODS):
                    # Mild winter seasonality, stronger on anti-infectives.
                    month = int(period[5:])
                    amplitude = 0.22 if p["atc1"].startswith("J") else 0.07
                    seasonal = 1.0 + amplitude * math.cos((month - 1) * math.pi / 6)
                    noise = rng.gauss(1.0, 0.09)
                    units = (p["_base"] * country_weight * channel_weight
                             * (1 + p["_trend"]) ** i * seasonal * noise)
                    if units <= 0:
                        continue
                    units = round(units)
                    rows.append({
                        "period": period,
                        "year": int(period[:4]),
                        "quarter": f"{period[:4]}Q{(month - 1) // 3 + 1}",
                        "country": country,
                        "region": region,
                        "sub_region": sub_region,
                        "atc1": p["atc1"],
                        "atc2": p["atc2"],
                        "atc3": p["atc3"],
                        "atc4": p["atc4"],
                        "molecule": p["molecule"],
                        "corporation": p["corporation"],
                        "manufacturer": p["manufacturer"],
                        "product": p["product"],
                        "form": p["form"],
                        "strength": p["strength"],
                        "pack_size": p["pack_size"],
                        "channel": channel,
                        "units": units,
                        "sales_eur": round(units * p["_price"] * rng.uniform(0.95, 1.05), 2),
                    })

    return pd.DataFrame(rows)


def main() -> None:
    df = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Written through DuckDB rather than pandas so the project needs no pyarrow:
    # DuckDB is already a dependency and carries its own Parquet writer.
    duckdb.sql("COPY df TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", params=[str(OUT)])
    size_mb = OUT.stat().st_size / 1e6
    print(f"{len(df):,} rows -> {OUT} ({size_mb:.1f} MB)")
    print(f"{df['molecule'].nunique()} molecules, {df['product'].nunique()} presentations, "
          f"{df['country'].nunique()} countries, {df['corporation'].nunique()} corporations, "
          f"{df['period'].nunique()} months")


if __name__ == "__main__":
    main()
