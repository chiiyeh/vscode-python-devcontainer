# Setting up

## Linking to the new repository

```bash
git remote set-url YOUR_REPO
```

You may want to delete the current git history.

## In devcontainer

Run

```bash
pipenv lock
```

This is done to generate the lock file.

Select pipenv as the python interpreter.
