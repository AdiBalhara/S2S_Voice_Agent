"""
Checks whether your Supabase connection works.

Reads these values from a .env file (project folder or my_agent folder):
  SUPABASE_URL     e.g. https://abcdxyz.supabase.co      (Project Settings -> API)
  SUPABASE_KEY     the anon key (or service_role key)    (Project Settings -> API)
  SUPABASE_TABLE   (optional) a table name to read 1 row from
  DATABASE_URL     (optional) the Postgres connection string
                   (Project Settings -> Database -> Connection string)

Run:  python test_supabase.py
"""
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()                      # .env in the current folder
load_dotenv("my_agent/.env")       # or the one inside my_agent

URL = os.getenv("SUPABASE_URL", "").rstrip("/")
KEY = os.getenv("SUPABASE_KEY", "")
TABLE = os.getenv("SUPABASE_TABLE", "")
DB_URL = os.getenv("DATABASE_URL", "")

results = []


def report(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


# ---- Test 1: is the project reachable and is the key accepted? ----
if URL and KEY:
    try:
        r = httpx.get(f"{URL}/auth/v1/health", headers={"apikey": KEY}, timeout=10)
        report("API reachable + key accepted", r.status_code == 200, f"HTTP {r.status_code}")
    except Exception as e:
        report("API reachable + key accepted", False, str(e))
else:
    print("[SKIP] API test - set SUPABASE_URL and SUPABASE_KEY in .env")

# ---- Test 2: read a row from your table through the Supabase client ----
if URL and KEY and TABLE:
    try:
        from supabase import create_client

        client = create_client(URL, KEY)
        data = client.table(TABLE).select("*").limit(1).execute().data
        report(f"Read from table '{TABLE}'", True, f"got {len(data)} row(s)")
    except Exception as e:
        report(f"Read from table '{TABLE}'", False, str(e))
elif URL and KEY:
    print("[SKIP] Table test - set SUPABASE_TABLE in .env to test reading data")

# ---- Test 3: direct Postgres connection ----
if DB_URL:
    try:
        import psycopg2

        conn = psycopg2.connect(DB_URL, connect_timeout=10)
        cur = conn.cursor()
        cur.execute("select now(), version();")
        now, version = cur.fetchone()
        conn.close()
        report("Direct Postgres connection", True, f"server time {now}")
    except Exception as e:
        report("Direct Postgres connection", False, str(e))
else:
    print("[SKIP] Postgres test - set DATABASE_URL in .env")

print()
if not results:
    print("Nothing was tested. Add your Supabase details to .env first.")
    sys.exit(1)
print("Connection is working." if all(results) else "Something failed - read the FAIL lines above.")
sys.exit(0 if all(results) else 1)