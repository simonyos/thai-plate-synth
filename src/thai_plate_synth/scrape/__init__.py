"""Scrape real Thai-plate images from public sources.

Images themselves are never committed — f0nt.com-style: the scraper code and
a `provenance.jsonl` audit trail are the only things that ship. Downstream
users re-run the scraper to reconstruct the benchmark locally.
"""
