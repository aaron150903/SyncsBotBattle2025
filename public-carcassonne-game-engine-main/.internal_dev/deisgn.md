# Design of Game
> DISCLAIMER: This document assumes that you have understood the game rules!

## Carcassonne Technicalities 
There are quite a few technicalities of the Carcassonne game. These include
1. A tile of the game will be considered a node (as each node has unique properties that we will look at later).
2. Each node will architecturally contain 4 vertices, 4 edges, a centre.
3. Each node will contain structure: monasteries, roads, cities (grass is a lack of a structure)
    - Structures can either be connection based or architecture based
3. Each node will contain `Structures` (connection-based) between vertices, and edges (the centre will not be use for connections).
    - These connections will represent cities, roads, as well as anything we may want to add in the extension pack.
    - Connections will be represented by the SuperClass `Structures`, and implment an interface `Connection`
    - Each connection will store 
4. The centre is primarily reserved for road start/stop points, and monastery `Structures`.
    - This allows for the engine to distinguish between road connections and road start and stop 
    - This allows for encoding of three way road start 
4. When a tile (or rather node) is placed, it must be place adjacent to another node's edge. 
    - This is just means that the game will not allow pieces to be placed diagonal to each other, in absence of any other adjacent piece.
6. To validate a node placed the game will check rules of both edge and vertex compatibility


Structure
- Monastery - always in the centre. If not using a centre always on the node itself.
- Cities - either one or two _internal_ vertex connections 
    - An _internal_ edge between two vertices is uniquely different than an _external_ edge between to vertices.
        - External: a node's walls. Primarily to validate node/tile placement. This is a programming abstraction, there is no actual in-game edge between the two vertices.
        - Internal: a unique connection between two edges. Represents a whole or part of a structure. Will primarily be referred to as a connection.
- Roads - Can be
    - a connection with two edges, 
    - A connection between the centre and (1 or 3 - this may change) Edges
    - edge case where there are two roads 

## Visualiser
