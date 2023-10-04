# Any Naive Bayes
Naive Bayes Classifier with Any Distribution

# Install

Please first install PDM >= 2.0 with pip/pipx.

```bash
pdm install --prod
```

# Develop

```bash
pdm install
```

# VSCode Settings

```bash
cp vscode_templates .vscode
```

Then install/activate all extensions listed in `.vscode/extensions.json`

# Usage

## API

```py
```

## Example

Check notebooks in the `examples` directory.

## Implement Custom Backend Distribution

This package currently only includes empirical distribution backed by scikit-learn's KDE. If you want to use other distributions you need to add custom wrapper class that implements `Distribution` abstract class. For more detail, please check `src/anynaivebayes/backends/kde.py` to understand how it is implemented. 