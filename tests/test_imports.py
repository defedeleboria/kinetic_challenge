"""Basic smoke‑test to ensure dependencies import without error."""

def test_streamlit_app_imports():
    import importlib
    modules = [
        "pandas",
        "joblib",
        "streamlit",
        "shap",
        "matplotlib.pyplot",
    ]
    for mod in modules:
        assert importlib.import_module(mod), f"Failed to import {mod}"
