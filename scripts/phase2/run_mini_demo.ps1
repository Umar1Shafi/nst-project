# scripts/phase2/run_mini_demo.ps1
# Run a tiny Phase-2 demo sweep on CPU, seed=77.
param(
  [string]$OutRoot = "out\phase2_demo"
)
$env:NST_SEED = "77"
python scripts\phase2\mini_sweep_runner.py `
  --contents "data\content\Still_Life.jpg" `
  --styles   "data\style\Monet.jpg" `
  --sw "15000,25000,30000" `
  --tv "0.0020,0.0032" `
  --edge "0.04" `
  --out-root $OutRoot `
  --device "cpu" --seed 77 --limit 3 --resume
