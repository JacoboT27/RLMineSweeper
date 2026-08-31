import glob, os, numpy as np

demo_dir = "demos"
files = sorted(glob.glob(os.path.join(demo_dir, "game_*.npz")))

games = len(files)
trans = wins = win_trans = 0
for f in files:
    d = np.load(f)
    n = len(d["act"])            # transitions in this game
    trans += n
    if int(d["won"]) == 1:
        wins += 1
        win_trans += n

print(f"games:            {games}")
print(f"  wins:           {wins}  ({wins/games:.0%})" if games else "  (no games yet)")
print(f"transitions:      {trans}")
print(f"  from wins only: {win_trans}")