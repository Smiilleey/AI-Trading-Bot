import time, argparse
from utils.approvals import _load, decide

def main(poll=2.0):
    print("Manual Approval Worker — polling queue. Ctrl+C to exit.")
    seen = set()
    while True:
        q = _load()
        for p in q["pending"]:
            if p["id"] in seen: continue
            seen.add(p["id"])
            print(f"\nREQUEST {p['id']} | {p['symbol']} | {p['side']} | lots={p['size_lots']} | stop={p['stop_pips']}")
            print(f"meta: {p.get('meta')}")
            ans = input("Approve? [y/N]: ").strip().lower()
            decide(p["id"], approve=(ans == "y"))
        time.sleep(poll)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll", type=float, default=2.0)
    args = parser.parse_args()
    main(args.poll)
