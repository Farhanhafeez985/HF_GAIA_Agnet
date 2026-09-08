import csv
from io import BytesIO
from sys import stderr
from traceback import print_exception
from zipfile import BadZipFile, ZipFile
import requests


def fetch_task_attachment(api_url: str, task_id: str) -> tuple[bytes, str]:
    """
    Returns (file_bytes, content_type) or (b'', '') if no attachment found.
    Follows any redirect the endpoint issues.
    """
    url = f"{api_url}/files/{task_id}"
    try:
        r = requests.get(url, timeout=15, allow_redirects=True)
    except requests.RequestException as e:
        print(f"[DEBUG] GET {url} failed → {e}")
        return b"", ""
    if r.status_code != 200:
        print(f"[DEBUG] GET {url} → {r.status_code}")
        return b"", ""
    return r.content, r.headers.get("content-type", "").lower()


def sniff_excel_type(blob: bytes) -> str:
    """
    Return one of 'xlsx', 'xls', 'csv', or '' (unknown) given raw bytes.
    """
    if blob[:4] == b"PK\x03\x04":
        try:
            with ZipFile(BytesIO(blob)) as zf:
                names = set(zf.namelist())
                if {"xl/workbook.xml", "[Content_Types].xml"} & names:
                    return "xlsx"
        except BadZipFile:
            pass  # fall through

    if blob[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        return "xls"
    try:
        sample = blob[:1024].decode("utf-8", "ignore")
        first_line = sample.splitlines()[0]
        if any(sep in first_line for sep in (",", ";", "\t")):
            csv.Sniffer().sniff(sample)
            return "csv"
    except (UnicodeDecodeError, csv.Error):
        pass

    return ""


def error_traceback(err: Exception, label: str = "") -> None:
    """
    Print the full stack trace of `err` to STDERR so it shows up in HF logs.
    """
    banner = f"[TRACE {label}]" if label else "[TRACE]"
    print(banner, file=stderr)
    print_exception(type(err), err, err.__traceback__, file=stderr)
    print("-" * 60, file=stderr)
