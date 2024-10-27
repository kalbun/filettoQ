from tensorflow import keras
from keras.layers import Dense
import numpy as np
import csv
import random
from typing import NewType, Tuple
from collections import deque

BOARD_CELLS = 9
STATES = 27
OFS_WHITE = 9
OFS_BLACK = 18
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
Elements 0:8 define the empty/filled state (0 = empty).
Elements 9:17 define if a white piece is present (1) or not (0).
Elements 18:26 do the same for black pieces.

"""
State = NewType('State',np.ndarray[STATES])

"""
Action describes the action to execute, using one-hot encoding.
Action[i] == 1 means that a piece will be put in position i.
"""
Action = NewType('Action', np.ndarray[ACTIONS])

def defineNetworks(_Q: keras.models.Sequential, _QStar: keras.models.Sequential):
    """
    Create network Q and initialise its weights with random values
    Repeat
        Generate some pairs x,y using it as a fitting model
        Train Q* with these values
        Copy Q* weights into Q weights
    Until the error goes below a certain threshold
    """
    print("Building neural networks")
    # create networks Q and Q*
    _Q.add(keras.Input(shape=STATES))
    _Q.add(keras.layers.Dense(32, activation='relu'))
    _Q.add(keras.layers.Dense(ACTIONS, activation='linear'))

    _QStar.add(keras.Input(shape=STATES))
    _QStar.add(keras.layers.Dense(32, activation='relu'))
    _QStar.add(keras.layers.Dense(ACTIONS, activation='linear'))

    # Q does not need loss function, we just it for predict
    _Q.compile()
    _QStar.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),loss=keras.losses.huber)

    for layer in _Q.layers:
        w: np.ndarray
        b: np.ndarray
        w,b = layer.get_weights()
        rng = np.random.default_rng()
        w = rng.standard_normal(size=(w.shape[0],w.shape[1]))
        b = rng.standard_normal(size=b.shape[0])
        layer.set_weights([w,b])

    # copy Q layers to Qstar
    for q_layer, qstar_layer in zip(_Q.layers, _QStar.layers):
        qstar_layer.set_weights(q_layer.get_weights())

def filetto(_state: State, action: Action, whiteMoves: bool ) -> Tuple[ State, int, bool]:
    """
    **Filetto simulator.**

    *Input*:
        state_s = current board state
        action_a = action to execute
        whiteMoves = True for white moves, False for black
    
    *Output*:
        A tuple containing:
        - new state s'
        - reward R(s)
        - a boolean True if the game is over, False otherwise
    """
    state: State = _state.copy()
    # Check if the move is valid (cell must be empty)
    if ( not action in range(STATES) and state[action] != 0 ):
        return state, REWARD_WRONG, False
    # Move is valid. Calculate reward. To do so, see if this move make us win or lose.
    state[action] = 1 # now occupied
    if (whiteMoves):
        state[action + 9] = 1
    else:
        state[action + 18] = 1
    #
    # Check winning combinations
    #
    for rowcol in range(2):
        if (np.dot(state[(rowcol*3):(rowcol*3)+3],np.ones(3)) == 3):
            # row is fully occupied, check white
            if (np.dot(state[(rowcol*3)+OFS_WHITE:(rowcol*3)+3+OFS_WHITE],np.ones(3)) == 3):
                # all pieces white, game won
                return state, REWARD_WIN, True
            if (np.dot(state[(rowcol*3)+OFS_BLACK:(rowcol*3)+3+OFS_BLACK],np.ones(3)) == 3):
                # all pieces black, game lost
                return state, REWARD_LOSE, True
        # columns are 0:3:6 or 1:4:7 or 2:5:8
        if (np.dot(state[[rowcol,rowcol+3,rowcol+6]],np.ones(3)) == 3):
            # column is fully occupied, check colors
            if (np.dot(state[[rowcol+OFS_WHITE,rowcol+3+OFS_WHITE,rowcol+6+OFS_WHITE]],np.ones(3)) == 3):
                # all pieces white, game won
                return state, REWARD_WIN, True
            if (np.dot(state[[rowcol+OFS_BLACK,rowcol+3+OFS_BLACK,rowcol+6+OFS_BLACK]],np.ones(3)) == 3):
                # all pieces black, game lost
                return state, REWARD_LOSE, True
    # check diagonals
    if (state[0] + state[4] + state[8] == 3):
        if (state[0+OFS_WHITE] + state[4+OFS_WHITE] + state[8+OFS_WHITE] == 3):
            # all pieces white, game won
            return state, REWARD_WIN, True
        if (state[0+OFS_BLACK] + state[4+OFS_BLACK] + state[8+OFS_BLACK] == 3):
            # all pieces black, game lost
            return state, REWARD_LOSE, True
    if (state[2] + state[4] + state[6] == 3):
        if (state[2+OFS_WHITE] + state[4+OFS_WHITE] + state[6+OFS_WHITE] == 3):
            # all pieces white, game won
            return state, REWARD_WIN, True
        if (state[2+OFS_BLACK] + state[4+OFS_BLACK] + state[6+OFS_BLACK] == 3):
            # all pieces black, game lost
            return state, REWARD_LOSE, True
    # check if the board is full with no winner
    if (np.dot(state[0:BOARD_CELLS],np.ones(ACTIONS)) == ACTIONS):
        return state, REWARD_OTHER , True
    return state, REWARD_OTHER , False

def train(_Q: keras.models.Sequential, _QStar: keras.models.Sequential) -> None:

    debug: bool = False
    if (debug == False):
        TRAIN_EPISODES = 200     # repetition of training with different minibuffers
        MINIBUFFER_SIZE = 32     # size of minibuffers
        EXPBUFFER_SIZE = 10000   # size of experience buffer during prediction phase
        SAMPLES_TO_PREDICT = 2000  # how many samples are create at each predict phase
        ACCEPTED_LOSS = 1e-3     # if average loss goes below this value, Q is declared trained
        LOSS_MAVG_SAMPLES = 250   # samples for calculating moving average of loss
        EACH_N_TRAINING = 25    # training cycles before print a dot during training
        EACH_N_PREDICTS = 25     # predictions before print a dot during predict
    else:
        TRAIN_EPISODES = 25     # at least 25
        MINIBUFFER_SIZE = 32
        EXPBUFFER_SIZE = 300
        SAMPLES_TO_PREDICT = 50
        ACCEPTED_LOSS = 100
        LOSS_MAVG_SAMPLES = 25
        EACH_N_TRAINING = 25
        EACH_N_PREDICTS = 25

    EPSILON: float = 1
    EPSILON_DECAY: float = 0.05
    GAMMA: float = 0.95
    state_s : State = np.zeros(STATES,dtype=np.int8)
    state_s1 : State = np.zeros(STATES,dtype=np.int8)
    action_a: int = 0
    isWhite: bool = True
    reward_in_s: float = 0.0
    return_for_s_a: float = 0.0
    is_game_over: bool = False

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
    csvfile.write("loss;")

    while (averageLoss > ACCEPTED_LOSS):

        training_cycle += 1
        print(f"Cycle {training_cycle}")
        # Build the experience buffer by accumulating samples
        print(f"\nPredict {SAMPLES_TO_PREDICT} samples (one dot = {EACH_N_PREDICTS} predictions)")
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
                # in case of random move, we must select a valid one by
                # checking if the corresponding cell in the board is empty
                while (state_s[action_a] != 0):
                    action_a = random.randrange(0,ACTIONS)
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

        print(f"\nTrain {TRAIN_EPISODES} times with {MINIBUFFER_SIZE} minibuffers (one dot = {EACH_N_TRAINING} training cycles)")
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
                currentLoss = _QStar.fit(x=QTrainX,y=QTrainY,verbose=0).history['loss'][0]
                lossBuffer.append(currentLoss)
            if ( (episode > 0) and (episode % EACH_N_TRAINING == 0)):
                # copy Qstar layers to Qs
                for q_layer, qstar_layer in zip(_Q.layers, _QStar.layers):
                    q_layer.set_weights(qstar_layer.get_weights())
                # update moving average
                averageLoss = np.average(list(lossBuffer)[-LOSS_MAVG_SAMPLES:])
                csvfile.write(f"{averageLoss}\n")
                print(f"Average loss: {averageLoss}")

    keras.models.save_model(_Q,"filetto_model_tf.bin",save_format="tf")
    csvfile.close()

def play(_Q: keras.models.Sequential):
    """
    Play filetto games until keyboard break
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
        # now we need to select only possible values. To do so, multiply by
        # the negate of board_state[0:9] which contains 1 if the cell is filled.
        returns_for_s = returns_for_s * (1 - board_state[0:BOARD_CELLS])
        # now get the position of maximum value: that's the move
        move = np.argmax(returns_for_s)
        return move

    # Start the game

    while (True):
        board = [' ' for _ in range(ACTIONS)]
        board_state: State = np.zeros(STATES,dtype=np.int8)
        whiteMoves: bool = True
        reward: float = 0.0
        isGameOver: bool = False
        draw_board(board)

        whiteMoves = True
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
    """
    Model Q is used to predict action with best return
    """
    Q = keras.models.Sequential()
    """
    Model QStar is trained with examples 
    """
    QStar = keras.models.Sequential()

    defineNetworks(Q,QStar)
    train(Q,QStar)
    play(Q)

main()