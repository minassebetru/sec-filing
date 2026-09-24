import os
import re
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": os.getenv(
        "SEC_USER_AGENT",
        "SEC Filing Research Assistant student-project@example.com",
    )
}


def normalize_cik(cik: str) -> str:
    return str(int(re.sub(r"\D", "", cik))).zfill(10)


def company_submissions(cik: str):
    cik = normalize_cik(cik)
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


def latest_filing(cik: str, form: str = "10-K"):
    data = company_submissions(cik)
    recent = data["filings"]["recent"]
    for i, filing_form in enumerate(recent["form"]):
        if filing_form == form:
            accession = recent["accessionNumber"][i]
            primary = recent["primaryDocument"][i]
            cik_num = str(int(normalize_cik(cik)))
            accession_no_dash = accession.replace("-", "")
            url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik_num}/"
                f"{accession_no_dash}/{primary}"
            )
            return {
                "company": data["name"],
                "cik": normalize_cik(cik),
                "form": form,
                "filing_date": recent["filingDate"][i],
                "accession": accession,
                "url": url,
            }
    raise ValueError(f"No recent {form} filing found for CIK {cik}")


def download_filing_text(filing):
    response = requests.get(filing["url"], headers=HEADERS, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)
