import csv
import os
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

INPUT_CSV   = "output_draft/kalshi_all_prefixes.csv"
OUTPUT_CSV  = "output_draft/domain_assignments.csv"

FOCUS_DOMAINS = {
    "Economics / Finance", "Politics / Government",
    "Weather", "Entertainment / Awards",
}

#EURUSD and KXEURUSD are same  (KALSHI change in format)

SPORTS = [
    "NFL", "NBA", "MLB", "NCAA", "NHL", "MLS",
    "PGA", "ATP", "WTA", "UFC", "NASCAR",
    "F1", "MVEN",
    "BUNDESLIGA", "EPL", "LALIGA",
    "LIGUE", "SERIEA", "UCL", "UEL",
    "EREDIVISIE", "LIGAMX", "LIGA",
    "BRASILIERO", "SUPERLIG",
    "WNBA", "INDY500", "LIVTOUR",
    "PICKLEBALL", "CSGO", "LOL", "CS2",
    "USOPEN", "THEOPEN", "MASTERS",
]

ECON = [
    "CPI",
    "FED", 
    "INX",
    "NASDAQ",
    "GDP",
    "WTI",
    "EUR",
    "USDJPY",
    "GBP",
    "TNOTE",
    "TSAW",
    "PAYROLL",
    "JOBLESS",
    "U3",
    "GAS",
    "GOLD",
    "PCE",
    "ADP",
    "ISMPMI",
    "BANKRUPT",
    "CRED",
    "FRM",
]

POLITICS = [
    "TRUMP", "BIDEN", "KAMALA", "VANCE",
    "538",
    "APRPOTUS",
    "DJT",
    "POWELL", "JPOW",
    "LEAVITT",
    "BERNIE", "ZELENSKY",
    "BILATERAL", "STARMER", "MAMDANI",
    "VAXX", "ASYLUM",
    "GOV",
]

WEATHER = [
    "HIGH",
    "RAIN",
    "TORNADO",
    "SNOW",
    "HURCAT",
    "FLIGHT",
    "CITIESWEATHER",
]

ENTERTAINMENT = [
    "GRAM",
    "OSCAR",
    "NETFLIX",
    "SPOTIFY",
    "TOPSONG",
    "TOPALBUM",
    "BILLBOARD",
]


def normalize(prefix):
    return prefix[2:] if prefix.startswith("KX") else prefix


def assign_domain(prefix):
    norm = normalize(prefix)

    if any(norm.startswith(p) for p in SPORTS):
        return "Sports"

    #FEDMENTION in Economics
    if any(norm.startswith(p) for p in ECON):
        return "Economics / Finance"

    #MENTION refers to  TRUMPMENTION, BIDENMENTION, POWELLMENTION..
    if "MENTION" in norm:
        return "Politics / Government"

    if any(norm.startswith(p) for p in POLITICS):
        return "Politics / Government"

    if any(norm.startswith(p) for p in WEATHER):
        return "Weather"

    if any(norm.startswith(p) for p in ENTERTAINMENT):
        return "Entertainment / Awards"

    return "Other"

def load_prefixes():
    if not os.path.exists(INPUT_CSV):
        sys.exit(f"Missing {INPUT_CSV}: (complete extract_cat.. first)")

    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def save_assignments(assignments):
    focus = [(p, d) for p, d in assignments if d in FOCUS_DOMAINS]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prefix", "domain"])
        writer.writerows(focus)


if __name__ == "__main__":
    prefixes = load_prefixes()

    assignments = [(row["prefix"], assign_domain(row["prefix"])) for row in prefixes]
    counts = Counter(domain for _, domain in assignments)
    
    save_assignments(assignments)
