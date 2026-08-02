import sys
import os
import json
import urllib.request

def check_mp_web(mp_id):
    url = f"https://materialsproject.org/materials/{mp_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req) as resp:
            html = resp.read().decode("utf-8")
            print(f"[{mp_id}] HTML response status 200, length: {len(html)} bytes.")
            # Check for space group or formula in html text
            if "BaSrI" in html:
                print(f"  [{mp_id}] Found formula BaSrI in HTML page!")
            if "Ca2Mg" in html:
                print(f"  [{mp_id}] Found formula Ca2Mg in HTML page!")
            return True
    except Exception as e:
        print(f"[{mp_id}] Web request error: {e}")
        return False

def main():
    target_ids = ["mp-754388", "mp-1178504", "mp-1949090", "mp-1039332"]
    print("Checking live Materials Project web pages directly...", flush=True)
    for mp_id in target_ids:
        check_mp_web(mp_id)

if __name__ == "__main__":
    main()
