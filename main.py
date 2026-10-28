"""Run the full pipeline: validate/load graph once, run algorithms, compare.

  python main.py                          # all three algorithms + comparison
  python main.py --algorithms louvain     # selected algorithms (louvain fast_greedy bldlp)
  python main.py --compare-only           # only regenerate comparison from saved results
  python main.py --skip-existing          # don't rerun algorithms that already have results
"""
import argparse
import traceback

import bldlp
import common
import compare_results
import fast_greedy
import louvain

RUNNERS = {
    "louvain": ("Louvain", lambda ds, a, log: louvain.run(ds, a.seed, log)),
    "fast_greedy": ("Fast_Greedy", lambda ds, a, log: fast_greedy.run(ds, log)),
    "bldlp": ("BLDLP", lambda ds, a, log: bldlp.run(ds, a.seed, a.alpha, a.max_iter, log)),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algorithms", nargs="+", choices=list(RUNNERS), default=list(RUNNERS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=int, default=15, help="BLDLP weight constant")
    ap.add_argument("--max-iter", type=int, default=10000, help="BLDLP iteration cap")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("--no-compare", action="store_true")
    a = ap.parse_args()

    summary, failed = {}, []
    if not a.compare_only:
        ds = common.load_dataset(log=common.get_logger("data"))
        for key in a.algorithms:
            name, fn = RUNNERS[key]
            if a.skip_existing and (common.RESULTS_DIR / name / f"{name}_metrics.json").exists():
                print(f"[skip] {name}: results exist")
                continue
            log = common.get_logger(name, common.RESULTS_DIR / name)
            try:
                summary[name] = fn(ds, a, log)
            except Exception:
                failed.append(name)
                log.error("%s FAILED:\n%s", name, traceback.format_exc())
    if not a.no_compare:
        try:
            compare_results.main()
        except SystemExit as e:
            print(f"Comparison not produced: {e}")

    print("\n=== Summary ===")
    for n, m in summary.items():
        print(f"{n:12s} communities={m['num_communities']:4d} modularity={m['overall_modularity']:.4f} "
              f"time={m['execution_time_seconds']:.2f}s -> results/{n}/")
    if failed:
        print("FAILED:", ", ".join(failed))


if __name__ == "__main__":
    main()
