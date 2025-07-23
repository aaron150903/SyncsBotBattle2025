Source code is in bot-algo.py:

1. Our bot smartly creates a copy of the grid and evaluates the move after placing the tile and running BFS to return structure_type, size and if it will be complete from the move and uses some of this information to determine best possible tile-placement. 

Our bot used a point based system when evaluating tile placements: We add bonuses if it completes our own structures. Bonuses for cities are also added for shield tiles which we increase in weight later on the game. We also add bonuses if we can find a large unclaimed piece of land or a bove where we can eventually steal from a player if the structure is near in proximity. While, we add penalties if we see the move helps complete enemy structures. 

If incomplete, we use probability by seeing how the minimum amount of valid tiles which can be used to complete structure. If we find the probability is impossible of completing the structure from the move we are about to make we will penalize it. 

For monastaries we place by first trying to find a location which can complete it as quick as possible (has most tiles around it), if we have one we reward based on how close the tile we have is close to finshing it as it frees up a meeple and a penalty for helping opponents complete their monastaries.

On top of this, we modify weights based on how much we are losing if we are losing we increase bonus multiplier and reduce penalty multiplier. Also increase bonus of completing later on game as value of completing becomes higher. 

2. Similarly, we evaluate the consequence of each potential move by creating a deep copy of the map and running BFS on each edge of the most recent tile since meeples can only be placed on that. Meeples are placed first seeing if we must address an opportunity to steal by completing our strategy of placing a tile which will eventually be connected. Then after this we evalate each placement based on potential points. This is again done by value in points * probability it can be completed. In addition, to this bonuses will be given to some structures depending on how aggresive our bot is based on competition.

3. This is efficient by score as we are evaluating all potential moves and their consequences. We value fairly based on the points a structure could bring us. We also eliminate moves which are unsensible based on the situation of the game by adjusting penalties and bonuses based on our performance relative to others and how late we are in the game. Also, make sure bonuses are enough to look for potential new opportunities to claim or connect land. In terms of time, we run each computation once and then pass it onto other functions to reduce time complexity. 