# Releasing

> ⚠️ **Both PyPI and TestPyPI are public, indexed registries.** Uploading there
> publishes the code/method to the world (a deleted version cannot be re-uploaded
> under the same number). The GitHub repo is currently **private**; only upload to
> the *real* PyPI once you intend the package to be public (e.g. on paper acceptance).
> TestPyPI below is a safe sandbox to rehearse the flow — but it is still public.

## 0. One-time tooling

```bash
python -m pip install --upgrade build twine "packaging>=24.2"
```

## 1. Build + validate (no account needed)

```bash
cd transcription-sync
rm -rf dist build src/*.egg-info
python -m build                 # -> dist/*.whl + dist/*.tar.gz
python -m twine check dist/*    # both must say PASSED
```

Already verified locally: `twine check` PASSED; the wheel installs in a clean
venv with **zero dependencies** (numpy is only needed for the `[audio]` extra).

## 2. TestPyPI account + token (manual — you do this once)

1. Create an account at <https://test.pypi.org/account/register/> (separate from PyPI).
2. Verify your email.
3. Create an API token: <https://test.pypi.org/manage/account/token/>
   — scope "Entire account" for the first upload (you can narrow it to the
   project after it exists). Copy the `pypi-...` token.

## 3. Upload to TestPyPI

Do **not** put the token in a file. Pass it at upload time:

```bash
python -m twine upload --repository testpypi dist/* \
    --username __token__ \
    --password 'pypi-AgEN...your-testpypi-token...'
```

(or set `TWINE_USERNAME=__token__` and `TWINE_PASSWORD=<token>` as env vars, or
let twine prompt you interactively).

On success twine prints the project URL:
`https://test.pypi.org/project/transcription-sync/0.1.0/`

## 4. Verify the published package installs from TestPyPI

TestPyPI doesn't host dependencies, so pull numpy from real PyPI:

```bash
python -m venv /tmp/ts-test && source /tmp/ts-test/bin/activate
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            "transcription-sync[audio]"
python -c "from transcription_sync import VisemeStream; print(len(VisemeStream().render_text('hello', amplitude=.3)), 'frames')"
deactivate
```

## 5. Real PyPI (later — only when going public)

Same as steps 2–3 but at <https://pypi.org> and `--repository pypi` (the default):

```bash
python -m twine upload dist/* --username __token__ --password 'pypi-...'
```

Remember to bump `version` in `pyproject.toml` and `__init__.py` for every
release — PyPI refuses to overwrite an existing version.
