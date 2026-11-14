# Notebooks

`substrax.notebooks` holds the tools behind the shared example guide
([Writing examples](../guides/writing-examples.md)). Install it with `substrax[notebooks]`;
`substrax-notebooks test` and the example tests also need pytest, `substrax[notebooks,testing]`.

| Module | Job |
| --- | --- |
| `substrax.notebooks.pairs` | `convert` writes an example's clean notebook; `check_pair` passes only when the committed notebook is exactly what `convert` writes |
| `substrax.notebooks.run` | `execute_example` runs a clean copy as a notebook from the repository root, within one budget, without the user's IPython profile, giving the example the `sys.path`, `sys.argv` and `__file__` a script run gives it, and failing if the run changes the repository; it reports each cell's outputs, the provenance and `main()`'s value |
| `substrax.notebooks.kernel_setup` | What a run gives the example's kernel before its first cell: a script run's `sys.path`, `sys.argv` and `__file__`, the guard that keeps `main()`'s value, the reporter, and the recorder IPython calls after each cell, which writes the qualified class of a cell's exception; sent as source, so the kernel never imports substrax |
| `substrax.notebooks.pages` | `generate_page` writes `docs/examples/<path>.md` from a run; `check_page` and `compare_with_run` check a committed page |
| `substrax.notebooks.conformance` | The guide's mechanical rules, by the ids the guide names them by |
| `substrax.notebooks.environment_writes` | `environment_writes` names the variables a statement writes into the process environment, for the `no-env-writes` rule and the source scans alike |
| `substrax.notebooks.corpus` | `corpus_problems` holds a repository's examples as a whole to the guide's corpus rules; `reaches_every_example` and `is_touched` decide which runs a change since a ref keeps |
| `substrax.notebooks.config` | The `[tool.substrax.examples]` settings |
| `substrax.notebooks.text` | `read_utf8`, the one way the tools read a script, notebook, page or hand-written document: UTF-8, a byte order mark dropped, every line ending `\n`, `NotUtf8Error` otherwise; `text_sha256`, the digest a page records of its script and its own text |
| `substrax.notebooks.repository` | The git queries the tools make |
| `substrax.notebooks.outputs` | `publish_outputs` makes a run's outputs fit to publish on a page; `local_paths` finds what still names a person's machine |
| `substrax.notebooks.terminal` | `strip_terminal_controls`, every terminal control sequence removed from what a page or a run error shows |
| `substrax.notebooks.cli` | `substrax-notebooks convert`, `check`, `page` and `test`; `python -m substrax.notebooks` runs it on the running interpreter |

The shared example tests that use them are `substrax.testing.example_suite` ([Testing](testing.md)).

::: substrax.notebooks.pairs

::: substrax.notebooks.run

::: substrax.notebooks.kernel_setup

::: substrax.notebooks.pages

::: substrax.notebooks.conformance

::: substrax.notebooks.environment_writes

::: substrax.notebooks.corpus

::: substrax.notebooks.config

::: substrax.notebooks.repository

::: substrax.notebooks.outputs

::: substrax.notebooks.terminal

::: substrax.notebooks.cli
