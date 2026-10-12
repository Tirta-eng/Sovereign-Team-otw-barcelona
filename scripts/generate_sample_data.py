"""Script to generate realistic sample/mock data for MTAH 2027.
Produces:
- data/tactical/BAR-ATH_tactical_atd.xml
- data/tactical/BAR-ATH_tactical_atd.txt
- data/tactical/BAR-ATH_tactical_smart_tagging.xml

Matches official Metrica FIFA/EPTS and SportsCode format.
"""

import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import games

def generate_sample_data():
    tactical_dir = os.path.join(ROOT, "data", "tactical")
    os.makedirs(tactical_dir, exist_ok=True)
    
    xml_path = os.path.join(tactical_dir, "BAR-ATH_tactical_atd.xml")
    txt_path = os.path.join(tactical_dir, "BAR-ATH_tactical_atd.txt")
    smart_path = os.path.join(tactical_dir, "BAR-ATH_tactical_smart_tagging.xml")
    
    print("1. Creating ATD metadata XML...")
    # 46 track slots: 23 Home, 23 Away
    players_xml = []
    for i in range(1, 47):
        team_id = "HomeTeam" if i <= 23 else "AwayTeam"
        players_xml.append(f'      <Player id="{i}" teamId="{team_id}"><Name>Track {i}</Name><ShirtNumber>{i}</ShirtNumber></Player>')
    
    xml_content = f"""<?xml version="1.0" encoding="utf-8"?>
<Metadata>
  <PitchDimensions>
    <FieldLength>105.3</FieldLength>
    <FieldWidth>68.0</FieldWidth>
  </PitchDimensions>
  <Teams>
    <Team id="HomeTeam">
      <Name>Barcelona</Name>
    </Team>
    <Team id="AwayTeam">
      <Name>Athletic Club</Name>
    </Team>
  </Teams>
  <Players>
{chr(10).join(players_xml)}
  </Players>
</Metadata>
"""
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_content)
    print(f"   Created {xml_path}")

    print("2. Creating Smart Tagging XML...")
    smart_xml = """<?xml version="1.0" encoding="utf-8"?>
<file>
  <ALL_INSTANCES>
    <instance>
      <start>447.0</start>
      <end>452.0</end>
      <code>SET PIECES</code>
      <label>
        <group>Type</group>
        <text>Kick Off (Start)</text>
      </label>
      <label>
        <group>Team</group>
        <text>Barcelona</text>
      </label>
    </instance>
    <instance>
      <start>4549.0</start>
      <end>4554.0</end>
      <code>SET PIECES</code>
      <label>
        <group>Type</group>
        <text>Kick Off (Start)</text>
      </label>
      <label>
        <group>Team</group>
        <text>Athletic Club</text>
      </label>
    </instance>
  </ALL_INSTANCES>
</file>
"""
    with open(smart_path, "w", encoding="utf-8") as f:
        f.write(smart_xml)
    print(f"   Created {smart_path}")

    print("3. Creating sample ATD raw tracking data (199,319 frames)...")
    gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    gt_df = pd.read_csv(gt_path)
    labels = gt_df["label"].values
    n_frames = len(labels)

    # Base coordinates for 46 slots (normalized 0-1)
    # Home: center around x=0.6, y=0.5; Away: center around x=0.4, y=0.5
    np.random.seed(42)
    home_base_x = np.random.uniform(0.45, 0.85, 23)
    home_base_y = np.random.uniform(0.15, 0.85, 23)
    away_base_x = np.random.uniform(0.15, 0.55, 23)
    away_base_y = np.random.uniform(0.15, 0.85, 23)

    # Write in chunks for high performance
    chunk_size = 10000
    with open(txt_path, "w", encoding="utf-8", buffering=1024*1024) as f:
        for chunk_start in range(0, n_frames, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_frames)
            lines = []
            for i in range(chunk_start, chunk_end):
                frame_id = i + 1  # 1-indexed in ATD
                label = labels[i]
                
                # Small jitter
                h_x = np.clip(home_base_x + np.random.normal(0, 0.01, 23), 0.05, 0.95)
                h_y = np.clip(home_base_y + np.random.normal(0, 0.01, 23), 0.05, 0.95)
                h_s = np.clip(np.random.normal(2.5, 0.8, 23), 0.1, 7.0)

                a_x = np.clip(away_base_x + np.random.normal(0, 0.01, 23), 0.05, 0.95)
                a_y = np.clip(away_base_y + np.random.normal(0, 0.01, 23), 0.05, 0.95)
                a_s = np.clip(np.random.normal(2.5, 0.8, 23), 0.1, 7.0)

                # Determine ball position
                if label == "H":
                    # Near player 0 (Home)
                    ball_x = h_x[0] + np.random.normal(0, 0.003)
                    ball_y = h_y[0] + np.random.normal(0, 0.003)
                    ball_s = np.random.uniform(3.0, 8.0)
                    ball_str = f"{ball_x:.4f},{ball_y:.4f},{ball_s:.2f}"
                elif label == "A":
                    # Near player 23 (Away, first away player)
                    ball_x = a_x[0] + np.random.normal(0, 0.003)
                    ball_y = a_y[0] + np.random.normal(0, 0.003)
                    ball_s = np.random.uniform(3.0, 8.0)
                    ball_str = f"{ball_x:.4f},{ball_y:.4f},{ball_s:.2f}"
                elif label == "D":
                    # Dead ball - either stationary on side or missing (NaN)
                    if np.random.rand() < 0.5:
                        ball_str = "NaN,NaN,NaN"
                    else:
                        ball_str = "0.5000,0.0100,0.00"
                else:
                    # Outside in-play / X
                    if np.random.rand() < 0.6:
                        ball_str = "NaN,NaN,NaN"
                    else:
                        ball_str = f"{np.random.uniform(0.3, 0.7):.4f},{np.random.uniform(0.3, 0.7):.4f},1.50"

                # Format players
                player_parts = []
                for p in range(23):
                    player_parts.append(f"{h_x[p]:.4f},{h_y[p]:.4f},{h_s[p]:.2f}")
                for p in range(23):
                    player_parts.append(f"{a_x[p]:.4f},{a_y[p]:.4f},{a_s[p]:.2f}")

                players_str = ";".join(player_parts)
                lines.append(f"{frame_id}:{players_str}:{ball_str}\n")
            
            f.writelines(lines)
            if (chunk_end // chunk_size) % 5 == 0 or chunk_end == n_frames:
                print(f"   Generated {chunk_end:,}/{n_frames:,} frames...")

    print(f"   Created {txt_path} successfully!")

if __name__ == "__main__":
    generate_sample_data()
