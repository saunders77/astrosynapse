import argparse

from astro2.autopilot_matches import run_match

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    run_match(parser.parse_args().folder)
