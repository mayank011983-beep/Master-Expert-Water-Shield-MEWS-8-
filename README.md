# MEWS 8

MEWS 8 is a Flask image-sealing service. It applies a visible watermark, a distributed DCT-based invisible seal, a cross-block high-frequency geometric consensus mesh, and signed PNG provenance metadata.

## Run locally

Set a long, random signing key before starting the service.

```powershell
$env:MEWS8_SIGNING_KEY = "replace-this-with-a-long-random-secret"
python -m pip install -r requirements.txt
python app.py
```

## Deploy with Gunicorn

```bash
export MEWS8_SIGNING_KEY='replace-this-with-a-long-random-secret'
gunicorn --bind 0.0.0.0:8000 app:app
```

## Endpoints

`POST /process` accepts multipart fields named `image` and `watermark` and returns a sealed PNG.

`POST /verify` accepts multipart field `image` and returns verification results for the signed metadata, sealed pixels, and distributed DCT payload.

`GET /health` returns service status.

The service provides tamper evidence, not remote execution or attempt counting: a PNG is passive data and cannot observe an external editor. Any changed pixels, rewritten provenance metadata, damaged DCT seal, or broken geometric mesh is reported by `/verify`.
