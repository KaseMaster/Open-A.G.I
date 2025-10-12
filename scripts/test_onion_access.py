#!/usr/bin/env python3
"""
Test script to access the local onion service through Tor SOCKS proxy.
Reads the onion hostname from HiddenServiceDir (from torrc) or falls back to ./onion_service/hostname.
Requires requests and PySocks (requests[socks]).
"""

import os
import re
import sys
import time
import pathlib

def read_torrc_hidden_service_dir(torrc_path):
    try:
        with open(torrc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        m = re.search(r'^HiddenServiceDir\s+"?([^"\n]+)"?', content, re.MULTILINE)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None

def read_onion_hostname(hs_dir):
    hostname_path = os.path.join(hs_dir, 'hostname')
    with open(hostname_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def main():
    torrc_path = os.path.join(os.getcwd(), 'config', 'torrc')
    hs_dir = read_torrc_hidden_service_dir(torrc_path) or os.path.join(os.getcwd(), 'onion_service')
    print(f"[info] HiddenServiceDir: {hs_dir}")

    if not os.path.isdir(hs_dir):
        print(f"[error] HiddenServiceDir not found: {hs_dir}")
        sys.exit(2)

    hostname = read_onion_hostname(hs_dir)
    url = f"http://{hostname}/"
    print(f"[info] Onion hostname: {hostname}")
    print(f"[info] Testing URL: {url}")

    try:
        import requests
    except Exception as e:
        print(f"[error] requests module not available: {e}")
        sys.exit(3)

    proxies = {
        'http': 'socks5h://127.0.0.1:9050',
        'https': 'socks5h://127.0.0.1:9050',
    }

    try:
        start = time.time()
        resp = requests.get(url, proxies=proxies, timeout=45)
        elapsed = time.time() - start
        print(f"[success] Status: {resp.status_code}, elapsed: {elapsed:.2f}s")
        # print small snippet
        text = resp.text
        print("[content]", text[:200].replace('\n', ' '))
        sys.exit(0)
    except Exception as e:
        print(f"[fail] Could not access onion service: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()