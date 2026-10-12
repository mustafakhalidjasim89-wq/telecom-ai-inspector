# Telecom AI Inspector — Cloud

Deploy `app.py` on Streamlit Community Cloud using `requirements.txt`.

Set the following in the Streamlit app **Secrets** settings (never commit them to GitHub):

```toml
APP_PASSWORD = "replace-with-a-long-unique-password"
RUNPOD_BASE_URL = "https://YOUR-POD-ID-8000.proxy.runpod.net/v1"
RUNPOD_VLLM_API_KEY = "replace-with-your-vllm-server-key"
```

RunPod must be running, accessible via HTTPS, and configured to require the vLLM API key. The RunPod account API key is **not** the same as the vLLM key.

**Security:** This app has a simple shared-password gate only. It does not provide individual accounts, audit logs, corporate SSO, rate limiting, or robust tenant isolation. Do not use for sensitive company documents without organizational approval and stronger identity/network controls. Community Cloud may not be approved for corporate data. RunPod proxy endpoints must also be secured.

The local file cache is disabled in cloud mode. Streamlit sessions still process uploads and model responses in memory. Avoid sensitive test data until approvals are in place.
