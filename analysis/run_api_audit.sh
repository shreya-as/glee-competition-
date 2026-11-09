#!/usr/bin/env bash
# Rebuild the server-side (API history/replay) audit used in the revised paper.
# Run from the repository root. Inputs: final_logs/*_with_replays.json and logs/v*_games.jsonl.
set -euo pipefail
mkdir -p analysis/out
F=final_logs
python3 analysis/extract.py "$F/sh_agent1_all_games_with_replays.json"       analysis/out/a1.jsonl
python3 analysis/extract.py "$F/shreyaAgent2_all_games_with_replays (3).json"   analysis/out/a2.jsonl
python3 analysis/extract.py "$F/Agent_3_all_games_with_replays.json"            analysis/out/a3.jsonl
python3 analysis/extract.py "$F/shreya_agent_4_all_games_with_replays.json"     analysis/out/a4.jsonl
python3 analysis/extract.py "$F/agent_5_all_games_with_replays.json"            analysis/out/a5.jsonl
python3 analysis/assign.py    # map each API game to a strategy version (logged game_id, else run window)
python3 analysis/metrics.py | tee analysis/out/metrics.txt   # per-window, per-version, per-family tables
python3 analysis/verify_claims.py | tee analysis/out/verify_claims.txt   # every number quoted in the paper
for f in "$F/sh_agent1_all_games_with_replays.json" "$F/shreya_agent_4_all_games_with_replays.json" \
         "$F/Agent_3_all_games_with_replays.json" "$F/agent_5_all_games_with_replays.json" \
         "$F/shreyaAgent2_all_games_with_replays (3).json"; do
  python3 analysis/inval.py "$f"   # invalid-move audit in the final window
done | tee analysis/out/invalid_moves.txt
for f in "$F/sh_agent1_all_games_with_replays.json" "$F/shreya_agent_4_all_games_with_replays.json" \
         "$F/Agent_3_all_games_with_replays.json" "$F/agent_5_all_games_with_replays.json" \
         "$F/shreyaAgent2_all_games_with_replays (3).json"; do
  python3 analysis/stale_pattern.py "$f"   # re-sent decisions among invalid moves
done | tee analysis/out/stale_pattern.txt
for f in "$F/sh_agent1_all_games_with_replays.json" "$F/shreya_agent_4_all_games_with_replays.json"; do
  python3 analysis/reject_share.py "$f"   # V9 vs V0 reject share in feasible W3 negotiation
done | tee analysis/out/reject_share.txt
python3 analysis/export_release.py   # writes the public paper_data/ folder (no message text)
python3 analysis/check_paper.py | tee analysis/out/check_paper.txt   # PASS/FAIL for every number in the paper
