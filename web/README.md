# Visual comparison

Open `index.html` or run `python -m http.server 8000 --directory web` from the project root. Our four matching-motion videos and posters are included. Author-hosted video players require a network connection. Third-party media is loaded from the original GitHub repositories; project links are also provided.

The local G1 policy is iteration 30000. Clips show only the first12 seconds, not a full-clip benchmark. Camera/robot differences prevent interpreting the page as a strict numerical ranking.

The dynamic robot/model/motion workbench is in `studio/`. Start it with `python scripts/visualize.py studio --port 8765`; a plain static server does not provide its evaluation and playback APIs.
