#!/usr/bin/env python3
"""Local E2E test: connect to JDownloader, crawl a FitGirl page, group by host, verify file picker."""
import os, sys, time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # env vars already set in Docker

from downloader import JDClient, detect_host, SUPPORTED_HOSTS

EMAIL = os.environ["MYJD_EMAIL"]
PASSWORD = os.environ["MYJD_PASSWORD"]
DEVICE = os.environ.get("MYJD_DEVICE", "JDownloader@oci")
TEST_URL = "https://fitgirl-repacks.site/the-witcher-3-wild-hunt-remastered/"

WARMUP = 10
GIVEUP_ZERO = 30

def main():
    print(f"1. Connecting to MyJDownloader (device={DEVICE})...")
    try:
        client = JDClient(EMAIL, PASSWORD, DEVICE)
        client.connect()
        print("   ✅ Connected")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        sys.exit(1)

    print(f"\n2. Starting crawl of {TEST_URL}")
    try:
        crawl_name = client.add_crawl([TEST_URL])
        print(f"   Crawl package: {crawl_name}")
    except Exception as e:
        print(f"   ❌ add_crawl failed: {e}")
        sys.exit(1)

    print("\n3. Polling crawl results...")
    stable, last_sup, last_total, found = 0, -1, -1, []
    for poll_n in range(60):
        time.sleep(3)
        try:
            found = client.crawl_links(crawl_name)
        except Exception as e:
            print(f"   poll {poll_n}: ERROR {e}")
            continue
        sup_count = sum(1 for l in found if detect_host(l.get("url", "")))
        print(f"   poll {poll_n}: total={len(found)} supported={sup_count}")

        if poll_n < WARMUP:
            last_sup, last_total = sup_count, len(found)
            continue

        if sup_count > 0 and sup_count == last_sup:
            stable += 1
            if stable >= 3:
                print(f"   ✅ Settled at poll {poll_n} with {sup_count} supported links")
                break
        elif sup_count == 0 and len(found) == last_total and poll_n >= GIVEUP_ZERO:
            print(f"   ⚠️ Gave up at poll {poll_n} with 0 supported")
            break
        else:
            stable = 0
        last_sup = sup_count
        last_total = len(found)
    else:
        print(f"   ⚠️ Hit 60-poll ceiling. supported={last_sup}")

    supported = [l for l in found if detect_host(l.get("url", ""))]
    if not supported:
        print("\n❌ No supported links found!")
        client.remove_crawl(crawl_name)
        sys.exit(1)

    print(f"\n4. Grouping {len(supported)} links by host...")
    groups: dict[str, list[dict]] = {}
    for l in supported:
        host = detect_host(l["url"])
        groups.setdefault(host, []).append({
            "url": l["url"],
            "name": l.get("name") or l["url"].rsplit("/", 1)[-1],
            "size": int(l.get("bytesTotal") or 0),
        })

    for host in sorted(groups):
        files = groups[host]
        total = sum(f["size"] for f in files)
        total_mb = total / (1024 * 1024) if total else 0
        print(f"   {host}: {len(files)} files ({total_mb:.0f} MB)")
        files.sort(key=lambda f: (0 if f["name"].endswith(".rar") else 1, f["name"]))
        for i, f in enumerate(files[:5], 1):
            sz = f" ({f['size'] / (1024*1024):.0f} MB)" if f["size"] else ""
            print(f"     {i}. {f['name'][:60]}{sz}")
        if len(files) > 5:
            print(f"     ... and {len(files) - 5} more")

    print(f"\n5. Cleaning up crawl from JD...")
    try:
        client.remove_crawl(crawl_name)
        print("   ✅ Crawl removed")
    except Exception as e:
        print(f"   ⚠️ remove_crawl: {e}")

    print(f"\n✅ Full flow verified! {len(supported)} links across {len(groups)} host(s).")
    print("   Hosts:", ", ".join(sorted(groups.keys())))

if __name__ == "__main__":
    main()
