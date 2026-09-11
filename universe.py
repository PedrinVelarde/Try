from __future__ import annotations

from typing import Dict, Optional

import pandas as pd


FALLBACK_SP500_STOCKS = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Alphabet": "GOOGL",
    "Amazon": "AMZN",
    "Meta": "META",
    "Nvidia": "NVDA",
    "AMD": "AMD",
    "Intel": "INTC",
    "Broadcom": "AVGO",
    "Tesla": "TSLA",
    "Netflix": "NFLX",
    "Adobe": "ADBE",
    "Oracle": "ORCL",
    "Salesforce": "CRM",
    "JPMorgan": "JPM",
    "Bank of America": "BAC",
    "Goldman Sachs": "GS",
    "Visa": "V",
    "Mastercard": "MA",
    "Morgan Stanley": "MS",
    "Coca-Cola": "KO",
    "PepsiCo": "PEP",
    "Walmart": "WMT",
    "Costco": "COST",
    "McDonald's": "MCD",
    "Nike": "NKE",
    "Caterpillar": "CAT",
    "Home Depot": "HD",
    "UnitedHealth": "UNH",
    "Exxon Mobil": "XOM",
    "Chevron": "CVX",
    "Eli Lilly": "LLY",
    "Johnson & Johnson": "JNJ",
    "Pfizer": "PFE",
    "Merck": "MRK",
    "Cigna": "CI",
    "AbbVie": "ABBV",
    "Danaher": "DHR",
    "Comcast": "CMCSA",
    "IBM": "IBM",
    "Honeywell": "HON",
    "Raytheon": "RTX",
    "Procter & Gamble": "PG",
    "3M": "MMM",
    "Amgen": "AMGN",
    "Bristol Myers Squibb": "BMY",
    "General Electric": "GE",
    "General Motors": "GM",
    "Wells Fargo": "WFC",
    "Cisco": "CSCO",
    "Verizon": "VZ",
    "AT&T": "T",
}


FALLBACK_EUROPEAN_STOCKS = {
    "ASML": "ASML",
    "SAP": "SAP",
    "LVMH": "LVMH",
    "TotalEnergies": "TTE",
    "Airbus": "AIR",
    "Sanofi": "SAN",
    "Novartis": "NVS",
    "AstraZeneca": "AZN",
    "Unilever": "UL",
    "HSBC": "HSBC",
    "Barclays": "BARC",
    "Shell": "SHEL",
    "BP": "BP",
    "Siemens": "SIE",
    "Schneider Electric": "SU",
    "Nokia": "NOK",
    "Dassault Systèmes": "DSY",
    "L'Oréal": "OR",
    "AIA Group": "AIA",
    "CRH": "CRH",
    "Diageo": "DEO",
    "Antofagasta": "ANTO",
    "BHP Group": "BHP",
    "Rio Tinto": "RIO",
    "Stellantis": "STLA",
    "Axa": "AXA",
    "Allianz": "ALV",
    "Münchener Rückversicherungs-Gesellschaft": "MUV2",
    "Volkswagen": "VOW3",
    "BMW": "BMW",
    "Mercedes-Benz": "MBG",
    "ING Group": "ING",
    "Banco Santander": "SAN",
    "UBS": "UBS",
    "Zurich Insurance Group": "ZURN",
    "Reckitt Benckiser": "RKT",
    "Adyen": "ADYEN",
    "Hermes": "RMS",
}


def load_current_sp500_symbols() -> Dict[str, str]:
    stock_map = {}

    try:
        current_tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        if current_tables:
            current_df = current_tables[0]
            symbol_column = next((col for col in ["Symbol", "Ticker", "Ticker symbol"] if col in current_df.columns), None)
            company_column = next((col for col in ["Security", "Company", "Name"] if col in current_df.columns), None)
            if symbol_column is not None and company_column is not None:
                filtered = current_df[[company_column, symbol_column]].dropna().copy()
                for _, row in filtered.iterrows():
                    company_name = str(row[company_column]).strip()
                    ticker = str(row[symbol_column]).strip()
                    if company_name and ticker:
                        stock_map[company_name] = ticker
    except Exception:
        pass

    return stock_map if stock_map else FALLBACK_SP500_STOCKS.copy()


def load_current_european_symbols() -> Dict[str, str]:
    return FALLBACK_EUROPEAN_STOCKS.copy()


def load_former_sp500_symbols() -> Dict[str, str]:
    stock_map = {}

    try:
        former_tables = pd.read_html("https://en.wikipedia.org/wiki/Former_components_of_the_S%26P_500")
        if former_tables:
            for former_df in former_tables:
                symbol_column = next((col for col in ["Symbol", "Ticker", "Ticker symbol"] if col in former_df.columns), None)
                company_column = next((col for col in ["Security", "Company", "Name"] if col in former_df.columns), None)
                if symbol_column is None or company_column is None:
                    continue

                filtered = former_df[[company_column, symbol_column]].dropna().copy()
                for _, row in filtered.iterrows():
                    company_name = str(row[company_column]).strip()
                    ticker = str(row[symbol_column]).strip()
                    if company_name and ticker:
                        stock_map[company_name] = ticker
    except Exception:
        pass

    return stock_map


def get_universe(
    include_former_constituents: bool = False,
    max_size: Optional[int] = None,
    include_current_only: bool = False,
    include_european: bool = False,
    include_sp500: bool = True,
) -> dict[str, str]:
    current = load_current_sp500_symbols()

    if include_sp500:
        merged = dict(current)
    else:
        merged = {}

    if include_former_constituents and include_sp500:
        former = load_former_sp500_symbols()
        merged = dict(former)
        merged.update(current)

    if include_european:
        merged.update(load_current_european_symbols())

    if include_current_only:
        return dict(current)

    if max_size is None or max_size <= 0:
        return merged

    items = list(merged.items())
    return dict(items[:max_size])
