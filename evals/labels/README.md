# Human labels

One worksheet per run, named `<run_id>.json`, created by:

    python -m evals.harness label <run_id>

Fill in `supported` and `specific`, then:

    python -m evals.harness calibrate <run_id>

These files are **committed on purpose**. They are the scarce artifact in the whole
eval setup -- the judge is replaceable, the hand labels are not -- and committing
them is what makes a calibration number reproducible by someone else.

Do not generate them programmatically. A synthetic label teaches the judge nothing
and quietly destroys the only independent signal here.
