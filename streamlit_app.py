# Entry point for Streamlit Cloud deployment
# This runs the frontend only (no FastAPI needed on cloud)
exec(open("app/frontend_standalone.py").read())
