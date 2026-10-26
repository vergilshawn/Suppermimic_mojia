# Publish

The `g1` and `mojia` branches share command names and repository structure. No GitHub remote or upload has been created automatically.

```bash
git remote add origin <your-repository-url>
git push -u origin g1
git push -u origin mojia
```

Data, checkpoints and logs are ignored. Publish chosen inference weights as Release attachments together with their SHA256 and `docs/MODELS.md`; keep multi-gigabyte optimizer/replay checkpoints local. Do not force-add `runs/` or entire datasets. See `THIRD_PARTY_NOTICES.md` for third-party data/media attribution.
