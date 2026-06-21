import yfinance as yf


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def format_percent(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"


def get_company_info(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        return info
    except Exception:
        return {}
