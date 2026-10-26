# A Curious Guide to AI Agents - companion code

Every code listing from the book, runnable, with tests. One file per listing; see [LISTINGS.md](LISTINGS.md)
for the section → file map.

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.11+
pip install -r requirements.txt
pytest -q                                              # ~50 tests, ~3 s, no API key needed
python -m ch05.l5_11_1_capstone                        # the complete agent + evals, offline
```

## Offline by default

Agent listings need a model. To keep the tests fast, free, and deterministic, they run against
`common/fakes.py`: a scripted chat model that plays back exact tool calls. That proves the parts *you* write -
tools, limits, approval pauses, tenant checks, eval scoring - but it does **not** prove a real model will choose
those tool calls. For that, run live:

```bash
pip install -r requirements-live.txt
cp .env.example .env   # add ANTHROPIC_API_KEY
export $(grep -v '^#' .env | xargs)
python -m ch05.l5_11_1_capstone --live     # runs the three evals against a real model
```

Model IDs in the listings (`anthropic:claude-sonnet-4-6`, `anthropic:claude-haiku-4-5`) are copied from the book.
Change them in one place per file if your account uses different ones.

## Where the code differs from the printed page

The book prints some listings as pseudocode or with assumed helpers ("`db` is your own database client").
Here those are filled in minimally so the listing runs, and the listing body itself is kept as close to the
printed text as possible. The helpers live in `common/` and `ch05/adapters.py`; they are stand-ins, not
production code. Each file's docstring says what was added.

## Versions

Pinned in `requirements.txt`; CI runs Python 3.11-3.13. The book targets LangChain/LangGraph 1.x; agent
frameworks move quickly, so if a listing breaks after an upgrade, check the pins first and open an issue.

## Exercises

`exercises/` holds the book's starters verbatim. Try them, then compare with `solutions/`.
To test your own version, change the import at the top of the matching test.
