"""spark-submit entrypoint: run one mbt training job on the driver.

Usage (issued by SparkComputeAdapter):

    spark-submit [...] job_wrapper.py <job.json>

The driver's Python environment must have mbt-core and the model's
training adapter installed.
"""

import sys


def main() -> int:
    from mbt.execute.job import main as run_job

    if len(sys.argv) != 2:
        sys.stderr.write("usage: spark-submit job_wrapper.py <job.json>\n")
        return 2
    return run_job([sys.argv[1]])


if __name__ == "__main__":
    raise SystemExit(main())
