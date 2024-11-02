from tensorflow import keras
from keras.layers import Dense
import numpy as np
import argparse
import random
from typing import NewType, Tuple
from collections import deque
import datetime
import wandb
from wandb.integration.keras import WandbMetricsLogger,WandbModelCheckpoint

BOARD_CELLS = 9
STATES = 18
OFS_WHITE = 0
OFS_BLACK = 9
ACTIONS = BOARD_CELLS

REWARD_WIN = 0.5
REWARD_LOSE = -0.5
REWARD_OTHER = 0
REWARD_WRONG = -1

"""
Let's define board's positions as follow:

    0   1   2
    3   4   5
    6   7   8

State is a vector defining the board state using one-hot coding.
Elements 0:8 define if a white piece is present (1) or not (0).
Elements 9:17 do the same for black pieces.

"""
State = NewType('State',np.ndarray[STATES])

"""
Action describes the action to execute, using one-hot encoding.
Action[i] == 1 means that a white piece will be put in position i
and a black piece in position i+9.
"""
Action = NewType('Action', np.ndarray[ACTIONS])

def defineNetworks() -> Tuple[keras.models.Sequential, keras.models.Sequential]:
    """
    Create two identical networks (Q and QStar) used during reinforcement
    learning operations.
    If model file <filetto_model_tf.bin> is found and coherent with Q setup,
    then Q is initialised with that file and then copied to Q*.
    Otherwise, weights are initialised to random values, identical for Q and Q*.

    Note that the file name is static. It is easy to make it configurable with
    a parameter, maybe later.

    *return*:
        Tuple contaning the two initialised and compiled networks
    """
    _QStar = keras.models.Sequential()
    _QStar.add(keras.Input(shape=STATES))
    _QStar.add(keras.layers.Dense(36, activation='relu'))
    _QStar.add(keras.layers.Dense(18, activation='relu'))
    _QStar.add(keras.layers.Dense(ACTIONS, activation='linear'))
    _QStar.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),loss=keras.losses.mean_squared_error)

    wandb.init(
        # set the wandb project where this run will be logged
        project="filettoQ",
        # track hyperparameters and run metadata with wandb.config
        config={
            "layer_1": 36,
            "activation_1": "relu",
            "layer_2": 18,
            "activation_2": "relu",
            "layer_3": 9,
            "activation_3": "linear",
            "optimizer": "Adam",
            "loss": "mean_squared_error",
            "metric": "accuracy",
            "epoch": 1,
            "batch_size": 32
        }
    )

    try:
        print("Loading neural networks")
        # load and compile
        _Q = keras.models.load_model("filetto_model_tf.bin",compile=True)
        # copy Q layers to Qstar
        for q_layer, qstar_layer in zip(_Q.layers, _QStar.layers):
            qstar_layer.set_weights(q_layer.get_weights())
    except:
        print("Building neural networks")
        # create networks Q and Q*
        _Q = keras.models.Sequential()
        _Q.add(keras.Input(shape=STATES))
        _Q.add(keras.layers.Dense(36, activation='relu'))
        _Q.add(keras.layers.Dense(18, activation='relu'))
        _Q.add(keras.layers.Dense(ACTIONS, activation='linear'))
        # Q does not need loss function, we just it for predict
        _Q.compile()
        for layer in _Q.layers:
            w: np.ndarray
            b: np.ndarray
            w,b = layer.get_weights()
            rng = np.random.default_rng()
            w = rng.standard_normal(size=(w.shape[0],w.shape[1]))
            b = rng.standard_normal(size=b.shape[0])
            layer.set_weights([w,b])


    return _Q,_QStar

def filetto(_state: State, action: Action, whiteMoves: bool ) -> Tuple[ State, float, bool]:
    """
    **Filetto simulator.**

    *Input*:
        state_s = current board state
        action_a = action to execute
        whiteMoves = True for white moves, False for black
    
    *Output*:
        A tuple containing:
        - new state s'
        - reward in s R(s)
        - game state, True if the game is over, False otherwise
    """
    state: State = _state.copy()
    # Check if the move is valid (cell must be empty)
    if ( 
        not action in range(ACTIONS)
        or
        state[action + OFS_WHITE] != 0
        or
        state[action + OFS_BLACK] != 0 
    ):
        return state, REWARD_WRONG, False
    # Move is valid. Calculate reward.
    if (whiteMoves):
        state[action + OFS_WHITE] = 1
    else:
        state[action + OFS_BLACK] = 1
    # Check winning combinations in rows and columns
    for rowcol in range(3):
        if (np.dot(state[(rowcol*3)+OFS_WHITE:(rowcol*3)+3+OFS_WHITE],np.ones(3)) == 3):
            # all pieces white, game won
            return state, REWARD_WIN, True
        if (np.dot(state[(rowcol*3)+OFS_BLACK:(rowcol*3)+3+OFS_BLACK],np.ones(3)) == 3):
            # all pieces black, game lost
            return state, REWARD_LOSE, True
        # columns are 0:3:6 or 1:4:7 or 2:5:8
            # column is fully occupied, check colors
        if (np.dot(state[[rowcol+OFS_WHITE,rowcol+3+OFS_WHITE,rowcol+6+OFS_WHITE]],np.ones(3)) == 3):
            # all pieces white, game won
            return state, REWARD_WIN, True
        if (np.dot(state[[rowcol+OFS_BLACK,rowcol+3+OFS_BLACK,rowcol+6+OFS_BLACK]],np.ones(3)) == 3):
            # all pieces black, game lost
            return state, REWARD_LOSE, True
    # check diagonals
    if (state[0+OFS_WHITE] + state[4+OFS_WHITE] + state[8+OFS_WHITE] == 3):
        # all pieces white, game won
        return state, REWARD_WIN, True
    if (state[0+OFS_BLACK] + state[4+OFS_BLACK] + state[8+OFS_BLACK] == 3):
        # all pieces black, game lost
        return state, REWARD_LOSE, True
    if (state[2+OFS_WHITE] + state[4+OFS_WHITE] + state[6+OFS_WHITE] == 3):
        # all pieces white, game won
        return state, REWARD_WIN, True
    if (state[2+OFS_BLACK] + state[4+OFS_BLACK] + state[6+OFS_BLACK] == 3):
        # all pieces black, game lost
        return state, REWARD_LOSE, True
    # check if the board is full with no winner
    if (np.dot(
        (state[OFS_WHITE:OFS_WHITE+BOARD_CELLS] + state[OFS_BLACK:OFS_BLACK+BOARD_CELLS])
        ,np.ones(BOARD_CELLS)
    ) == BOARD_CELLS):
        return state, REWARD_OTHER , True
    return state, REWARD_OTHER , False

def train(_Q: keras.models.Sequential, _QStar: keras.models.Sequential, initial_epsilon:float = 1) -> None:
    """
    Perform the training of Q network

    *Input:*
        _Q = the Q network to train
        _QStar = second neural network used in DQN. Strictly speaking, it could be defined
        inside the function, because it is never used elsewhere. But for the moment, Q* is
        defined in the initialisation function. Let it stay as is.
        initial_epsilon = starting value for epsilon-greedy policy, default 1. Changing the
        starting value of epsilon is important if you load an existing model already reliable
        and want to reduce random exploration.
    """

    # Set debug to true if you want to perform very quick predict/fit cycles to check if
    # the code contains errors.
    debug: bool = False
    if (debug == False):
        TRAIN_EPISODES = 250        # repetition of training with different minibuffers
        MINIBUFFER_SIZE = 32        # size of minibuffers
        EXPBUFFER_SIZE = 10000      # size of experience buffer during prediction phase
        SAMPLES_TO_PREDICT = 1000   # how many samples are create at each predict phase
        ACCEPTED_LOSS = 0.01        # if average loss goes below this value, Q is declared trained
        LOSS_MAVG_SAMPLES = 100     # samples for calculating moving average of loss
        EACH_N_TRAINING = 50        # training cycles before print a dot during training
        EACH_N_PREDICTS = 50        # predictions before print a dot during predict
    else:
        TRAIN_EPISODES = 25         # at least equal to EACH_N_TRAINING 
        MINIBUFFER_SIZE = 32
        EXPBUFFER_SIZE = 300
        SAMPLES_TO_PREDICT = 50
        ACCEPTED_LOSS = 100
        LOSS_MAVG_SAMPLES = 25
        EACH_N_TRAINING = 25
        EACH_N_PREDICTS = 25

    EPSILON: float = initial_epsilon
    EPSILON_DECAY: float = 0.01
    GAMMA: float = 0.95
    state_s : State = np.zeros(STATES,dtype=np.int8)
    state_s1 : State = np.zeros(STATES,dtype=np.int8)
    action_a: int = 0

    # Keeps the last loss values to calculate loss moving average
    lossBuffer = deque(maxlen=LOSS_MAVG_SAMPLES*4)
    # Create a fixed-size buffer that automatically discards oldest experiences when full
    experience_buffer = deque(maxlen=EXPBUFFER_SIZE)

    # start with initial state s
    # Predict the return values for s. The NN returns all the nine values.
    # Use epsilon-greedy policy to choose action that either maximizes
    # return, or is random.
    # call simulator to get reward(s) and state s' from state s and action a.
    # store s,a,s',reward and return into experience buffer
    # fit Q*, then copy its weights to Q.

    state_s = np.zeros_like(state_s)
    training_cycle: int = 0
    averageLoss: float = 1000.0

    csvfile = open("losses.csv","wt")
    csvfile.write("loss\n")

    while (averageLoss > ACCEPTED_LOSS):

        returns_for_s_a: np.ndarray = np.array(ACTIONS)
        return_for_s_a: float = 0.0
        isWhite: bool = True
        reward_in_s: float = 0.0
        is_game_over: bool = False

        training_cycle += 1
        print(f"Cycle {training_cycle}")
        print(f"\nPredict {SAMPLES_TO_PREDICT} samples (one dot = {EACH_N_PREDICTS} predictions)")
        print(f"epsilon is {EPSILON}")
        # Build the experience buffer by accumulating samples
        for sample in range(SAMPLES_TO_PREDICT):
            # Predict the return values for s. The NN returns all the nine values.
            # Use epsilon-greedy policy to choose action that either maximizes
            # return, or is random.
            returns_for_s_a = _Q.predict(x= state_s.reshape(1,-1),verbose=0)
            if (random.random() > EPSILON):
                # get the maximum return value. Subtract -10000 from return for filled
                # cells so that they are never selected
                return_for_s_a = np.max(returns_for_s_a[0] - (state_s[0:ACTIONS] * 10000 ))
            else:
                # in case of random move, limit the choice to empty cells
                action_a = np.random.choice(np.where(state_s[0:9]+state_s[9:18] == 0)[0])
                return_for_s_a = returns_for_s_a[0,action_a]
            # call simulator to get reward(s) and state s' from state s and action a.
            state_s1, reward_in_s, is_game_over = filetto(_state=state_s,action=action_a,whiteMoves=isWhite)
            return_for_s1_a1 = np.max(_Q.predict(x= state_s1.reshape(1,-1),verbose=0))
            # store s,a,s',reward and return into experience buffer
            experience_buffer.append([state_s,action_a,reward_in_s,return_for_s_a,state_s1,return_for_s1_a1])
            if (is_game_over):
                state_s = np.zeros_like(state_s)
            else:
                state_s = state_s1
            # alternate white and black moves (used in the simulator)
            isWhite = not isWhite
            if ((sample > 0) and (sample % EACH_N_PREDICTS == 0)):
                print(".",end="")
        EPSILON = max(EPSILON_DECAY,EPSILON-EPSILON_DECAY)

        print(f"\nTrain {TRAIN_EPISODES} times with minibuffers of {MINIBUFFER_SIZE} samples")
        # Now create random minibuffers to train the network
        for episode in range(1,TRAIN_EPISODES+1):
            print(".",end='')
            QTrainX: np.ndarray = np.zeros([MINIBUFFER_SIZE,STATES])
            QTrainY: np.ndarray = np.zeros([MINIBUFFER_SIZE,1])

            # Sample randomly from our experience buffer
            sample_size = min(MINIBUFFER_SIZE, len(experience_buffer))
            sampled_experiences = random.sample(list(experience_buffer), sample_size)
            for row in range(sample_size):
                # Apply Belman optimality equation to values from the experience buffer
                state_s,action_a,reward_in_s,return_for_s_a,state_s1,return_for_s1_a1 = sampled_experiences[row]
                QTrainX[row] = state_s.reshape(1,-1)
                # Mistake! return_for_s_a is left side of Belman's equation and not part of y
                # QTrainY[row] = (return_for_s_a + reward_in_s + GAMMA * return_for_s1_a1).reshape(1,-1)
                QTrainY[row] = (reward_in_s + GAMMA * return_for_s1_a1).reshape(1,-1)
            currentLoss = _QStar.fit(
                x=QTrainX,
                y=QTrainY,
                batch_size=MINIBUFFER_SIZE,
                verbose=0,
                callbacks=[
                    WandbMetricsLogger(log_freq=5),
                    WandbModelCheckpoint("models",verbose=0)
                ]
            ).history['loss'][0]
            lossBuffer.append(currentLoss)
            if ( (episode > 0) and (episode % EACH_N_TRAINING == 0)):
                # copy Qstar layers to Qs
                for q_layer, qstar_layer in zip(_Q.layers, _QStar.layers):
                    q_layer.set_weights(qstar_layer.get_weights())
                # update moving average
                averageLoss = np.average(list(lossBuffer)[-LOSS_MAVG_SAMPLES:])
                csvfile.write(f"{averageLoss}\n")
                print(f"Average loss: {averageLoss}")
                del QTrainX, QTrainY, sampled_experiences
        # save model at each complete iteration, this avoids losing work
        # if something goes wrong
        keras.models.save_model(_QStar,"filetto_model_tf.bin",save_format="tf")
        print("\nModel was saved in filetto_model_tf.bin")
        csvfile.flush()
    csvfile.close()

def play(_Q: keras.models.Sequential):
    """
    Play filetto games until CTRL-C is pressed.
    The function interactively draws the board and asks user for
    a move. To specify the move, user must press a number from 1
    to 9, then Enter. Invalid moves are blocked.
    In case of victory, loss or draw, the board is cleared and the
    game restarts. First mover, either human or PC, is randomly
    chosen at each game.
    The graphical aspect is very poor, but this was an exercise
    of DQN, not of style :-)

    Input:
        _Q = a (supposedly) trained network.
    """
    # Function to draw the board
    def draw_board(board):
        print('\n' + board[0] + '|' + board[1] + '|' + board[2])
        print('-----')
        print(board[3] + '|' + board[4] + '|' + board[5])
        print('-----')
        print(board[6] + '|' + board[7] + '|' + board[8] + '\n')

    # Function for the user's turn
    def user_turn(board):
        while True:
            move = input("Enter your move (1-9): ")
            if move.isdigit() and int(move) in range(1, 10) and board[int(move) - 1] == ' ':
                move = (int)(move) - 1
                return move
            else:
                print("Invalid move. Try again.")

    # Function for the computer's turn
    def computer_turn(board_state):
        # retrieve all the returns for all the moves
        returns_for_s = _Q.predict(x= board_state.reshape(1,-1),verbose=0)
        # Now get the position of maximum value, that's the move with highest return.
        # Note that, to avoid already filled cells, we set the corresponding
        # return value to a very low number.
        returns_for_s[0, board_state[0:9]+board_state[9:18] != 0 ] = -1e5
        move = np.argmax(returns_for_s)
        return move

    # Start the game
    while (True):
        board = [' ' for _ in range(ACTIONS)]
        board_state: State = np.zeros(STATES,dtype=np.int8)
        whiteMoves: bool
        reward: float = 0.0
        isGameOver: bool = False
        draw_board(board)

        # Randomly choose first mover
        whiteMoves = (random.random() >= 0.5)
        while True:
            if (whiteMoves):
                # User's turn
                move = user_turn(board)
                board_state,reward,isGameOver = filetto(board_state,move,whiteMoves)
                board[move] = 'O'
            else:
                move = computer_turn(board_state)
                board_state, reward, isGameOver = filetto(board_state,move,whiteMoves)
                board[move] = 'X'
            draw_board(board)
            whiteMoves = not whiteMoves   # change turn
            if (isGameOver):
                if (reward == REWARD_WIN):
                    print("You win!")
                elif (reward == REWARD_LOSE):
                    print("You lost!")
                else:
                    print("It's a draw!")
                break
            
def main():

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        "Filetto 0.1\nA silly game to learn DQN"
    )
    parser.add_argument('-l','--load',action='store_true',help="Load existing network model from filetto_model_tf.bin")
    parser.add_argument('-e','--epsilon',default=1.0,help="Set starting value for epsilon-greedy policy. Default 1.0")
    args = parser.parse_args()
    Q: keras.models.Sequential
    QStar: keras.models.Sequential

    if (args.epsilon < 0.01):
        print(f"{args.epsilon} is too low for epsilon, setting to 0.01")
        args.epsilon = 0.01
    if (args.epsilon > 1):
        print(f"{args.epsilon} is too big for epsilon, setting to 1.0")
        args.epsilon = 1.0
    Q,QStar = defineNetworks()
    if (args.load):
        train(Q,QStar,args.epsilon)
    play(Q)

main()