# Publish DeepSVerify

Status: source prepared; public repository and hosted URL are not created by this package.

## 1. Add the actual trained model

Open Command Prompt in the extracted folder containing `streamlit_app.py`. Use Python 3.12 and run:

```cmd
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python scripts\export_deploy_model.py --source "C:\YOUR\DeepSVerify_windows_demo\DeepSVerify\checkpoints\baseline_v1"
```

Replace the quoted path with your actual completed-demo folder. This creates a smaller `models/best.pt` plus a matching `models/threshold.json`. Do not copy the original 134 MB training checkpoint into Git: GitHub blocks ordinary files larger than 100 MiB. The inference-only state is expected to be about 43 MiB; the exporter prints the measured size. No retraining is needed.

Test locally:

```cmd
python -m streamlit run streamlit_app.py
```

## 2. Create a public GitHub repository

On GitHub, create a repository named `DeepSVerify`, select **Public**, and leave automatic README, license and gitignore creation unchecked because this folder already contains README and gitignore files. This package does not assign a license to datasets or third-party content.

Install Git for Windows if unavailable. Open Command Prompt here and run:

```cmd
git init -b main
git add .
git status
git commit -m "Publish DeepSVerify research demo and measured results"
git remote add origin https://github.com/YOUR_USERNAME/DeepSVerify.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username. Sign in through Git's browser prompt; never paste a password/token into chat. If Git asks for a commit name/email, configure your preferred identity locally, then retry the commit. Use Git push, not browser file upload, for the approximately 43 MiB model (browser uploads have a 25 MiB limit).

Before pushing, `git status` should include only project source, public aggregate results, assets and the compact model. Do not add datasets or personal signatures.

## 3. Deploy Streamlit

Visit https://share.streamlit.io/ and sign in with the GitHub account that owns the repository. Create an app with:

- Repository: `YOUR_USERNAME/DeepSVerify`
- Branch: `main`
- Main file: `streamlit_app.py`
- Advanced settings → Python: `3.12`
- Optional subdomain: choose any available name.
- No app secrets required.

Click Deploy and wait for the build to finish. If a build fails, copy the error from the deployment logs. Do not change the validation threshold to fix installation errors.

## 4. Verify the public link

Open the generated URL in a private browser window. Confirm the verification screen loads, the measured results tab displays the recorded metrics, and the trained model compares two images. Keep your existing Windows demo available as a presentation fallback.

Add the actual successful URL to the top of README.md and the GitHub repository's Website field. No live URL is supplied in advance.
