# Mosaic, the machine

`mosaic-machine.html` is a self-contained, offline figure: Mosaic drawn as a hand-inked machine running one real
question from the live Cerebras run (q-a-128k-009, "What was the total amount refunded for billing issues in May?").
Live at **https://zacharias1219.github.io/Mosaic/** (published by `.github/workflows/pages.yml` on every push to
`main`), or open the file in a browser. The buttons follow the real steps (ask, filter in code, send to the model, read the answer),
the Pipeline picker replays P3, P0f and P0 with their live answers, and See-through / Labels show the working parts.

- `mosaic-machine.src.html`: the figure's source: parts list, how each step really works, geometry, behaviour.
- `core.js`: the Colorfigures engine the source loads while editing (MIT, see `LICENSE-colorfigures`).

Rebuild the single file after editing the source, with the Colorfigures skill's build script:

```bash
python3 <colorfigures skill>/scripts/build.py mosaic-machine.src.html mosaic-machine.html
```
