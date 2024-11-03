**Preparation**

To run filettoQ, you need to install numpy and tensorflow.
If you want to get in-depth data analysis of your network,
you will also need wandb and a valid API key.

**Execution**

Run *python filettoQ.py* from command line or via Visual Studio or any other
development environment.
The application has a switch for initialising epsilon. This can be useful
in case you load the 

**Overall notes**

Due to inexperience, writing the code has been much more complex
than expected.
I did many mistakes, possibly all those in the "manual of DQN beginner":
- errors in the formula for calculating the return
- errors in the simulator
- network too large
- gamma decay too fast

I also implemented some improvements over time that helped DQN converge:
- minibuffers that proved to be quite a game changer.
- reward normalisation. Rather than putting, for example, +100 in case of loss,
    I set -1 for forbidden move, 0.5 for victory, -0.5 for loss, 0.0 otherwise.
    Too high rewards were possibly introducing instability.
- creation of an experience buffer storing 10000 samples
- Weights update every 32 training cycles rather than at every cycle.

I didn't use smooth weights update from Q* to Q, but the network converges.

**Problem definition**

Two players (white and black)
Board with nine positions
Each position can be empty or contain either piece

Let's define board's positions as follow:

    0   1   2
    3   4   5
    6   7   8

*State and Actions*

The board is completely defined with 18 elements, using one-hot coding.
If state[0:18] is the vector describing board state, and c is a cell, then:
    state[c] = 1 if there is a white piece, 0 otherwise
    state[c+9] = 1 if there is a black piece, 0 otherwise

*Definition of Q, Q*

Q and Q* input size is 18 and output size is 9.
This configuration comes from an Andrew Ng's suggestion (see Coursera
specialisation in ML) and makes Q output the return value for all
the nine moves. It is also possible to increase the input to 19 elements,
also passing action a, and getting only return(s,a).
Yet, it turns out that the chosen configuration is simpler to handle in code.

*Rewards*

We then assign:
-1 for forbidden moves, not used as illegal moves are blocked before execution
0.5 for victory (human)
-0.5 for loss
0 in all other cases

**Simulator**

The simulator receives the state s and an action a. It returns the new state S',
the reward R, a boolean True if the game is over.
If execution of a would bring to a forbidden configuration, then S' = S

filetto(S,A) -> S', R, gameover

**Playing filetto**

During the game, the computer uses Q.predict() to calculate the best from its
point of view, that is, the move with lowest return.
