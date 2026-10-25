# Redeploying the George Market Forecast Lab (Streamlit Community Cloud)

This folder is the complete app. It needs no model training: it reads the saved results in `data/`.

```
IFS742_Dashboard_Deploy/
├── app.py                  the dashboard
├── requirements.txt        packages Streamlit Cloud installs
├── .streamlit/config.toml  light theme (keeps all table text readable)
└── data/
    ├── predictions.csv     every model + both baselines, harmonised and as submitted
    └── monthly_prices.csv  approved FINAL v3 monthly prices
```

## Option A: update the group's existing app (keeps the same link)

Use this if you (or the member who deployed it) can edit the GitHub repository behind
`ifs742-overall-dashboard-…streamlit.app`.

1. Open the repository on github.com.
2. Click **Add file → Upload files**.
3. Drag in `app.py`, `requirements.txt`, the `data` folder and the `.streamlit` folder from this folder.
   GitHub replaces files with the same name.
   - Hidden folders like `.streamlit` may not show in File Explorer. In Explorer, choose **View → Show → Hidden items**.
   - If dragging the folder doesn't work, create the file on GitHub instead: **Add file → Create new file**, type the name `.streamlit/config.toml`, and paste in the contents of that file.
4. Click **Commit changes**.
5. Streamlit Cloud redeploys automatically within a minute or two. Open the app link and check it.
   - If the app runs a different main file (e.g. `streamlit_app.py`), either rename `app.py` to match before uploading, or change the main file in the app's settings on share.streamlit.io.

## Option B: deploy a new app from your own account (new link)

1. **Create a GitHub repository**
   - Sign in at github.com, click **New** (top left), name it `ifs742-forecast-lab`, choose **Public**, and click **Create repository**.
   - Click **uploading an existing file**, drag in everything in this folder (including `data` and `.streamlit`), and click **Commit changes**.
2. **Deploy on Streamlit**
   - Go to share.streamlit.io and sign in with GitHub.
   - Click **Create app → Deploy a public app from GitHub**.
   - Repository: `your-username/ifs742-forecast-lab` · Branch: `main` · Main file path: `app.py`.
   - Optional: under **App URL**, choose a name such as `ifs742-george-forecast-lab`.
   - Click **Deploy**. The first build takes 2–5 minutes.
3. **Update the report:** replace the Streamlit link on the first page of the group report with the new URL.

## Check it locally first (optional)

```
pip install -r requirements.txt
streamlit run app.py
```

## If something goes wrong

| Problem | Fix |
|---|---|
| "ModuleNotFoundError" in the Streamlit log | Make sure `requirements.txt` is in the repository's top folder. |
| "FileNotFoundError: data/predictions.csv" | The `data` folder wasn't uploaded, or was uploaded into a subfolder. It must sit next to `app.py`. |
| App shows a dark theme | `.streamlit/config.toml` is missing; create it on GitHub as described in Option A. |
| "This app has gone to sleep" | Free apps sleep after inactivity; click **Yes, get this app back up**. Open it once before the marker does. |
