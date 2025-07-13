# Scoring and Ranks
- Previously Bot Battle used win rates, with set interval round-robin game runs. I am not sure how I feel about this since, it is not robust to reactions.
    - It also does not allow for a different play-style that you could do with Carcassonne Round Robin - Play for points, more adversarial bots would be discouraged
    
- What I am thinking is Round Robin "Tournament" Should be held at a set interval
    - Overall Rank is a sliding window of the last 30 (not finalised) minutes (so if a game is every minute 30 roundrobin tournaments decide your rank)
    - Ideally the number of points you get should be _offset_ (but not weighted) but how difficult opponent bots are.
        - Offset = k1 * (your_score - agg_opp_mean) / (agg_opp_sd + ε) \
           + k2 * (your_mean - agg_opp_mean) / (your_sd + agg_opp_sd + ε) capped at [2, -4]

        - Win Bonus - +4 First, +2 for Second

> Definitely not fixed. The advantage of winrate is that it can be run at an arbitrary interval. So to be discussed.
