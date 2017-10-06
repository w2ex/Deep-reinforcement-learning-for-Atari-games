import sys
from ale_python_interface import ALEInterface
import tensorflow as tf
import cv2
import random
import numpy as np
from collections import deque


ale = ALEInterface()
ale.setInt( 'random_seed' , 123)

USE_SDL = True # to display the game
if USE_SDL:
    ale.setBool( 'sound' , False)
    ale.setBool( 'display_screen' , True)
           
ale.loadROM( 'Breakout.bin' )
legal_actions = ale.getMinimalActionSet()
ACTIONS = len(legal_actions) # number of legal actions
GAMMA = 0.99 # decreasing factors of previous observations
OBSERVE = 100000. # number of frames to observe before learning
EXPLORE = 1000000. # number of frames for epsilon to reach its final value
FINAL_EPSILON = 0.1 # epsilon final value
INITIAL_EPSILON = 1.0 # epsilon initial value, the probability to choose a random action
REPLAY_MEMORY = 100000 # number of last state transitions kept in memory
BATCH = 32 # minibatch size
K = 1 # select an action every K frames
OFFSET = 0 # offset to continue the learning after an interruption, to display properly graphs in TensorBoard
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01) # randomly initialize the weights to shape the two first layers
    return tf.Variable(initial)
    
def bias_variable(shape): # idem for the bias
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)
    
def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def createNetwork():
    # Network weights
    with tf.name_scope( 'Convolution_1_weights' ):
        W_conv1 = weight_variable([8, 8, 4, 32]) # defining the weights of the first convolution layer
    with tf.name_scope( 'Convolution_1_bias' ):
        b_conv1 = bias_variable([32])

    with tf.name_scope( 'Convolution_2_weights' ):
        W_conv2 = weight_variable([4, 4, 32, 64]) # idem for the second convolution layer
    with tf.name_scope( 'Convolution_2_bias' ):
        b_conv2 = bias_variable([64])

    with tf.name_scope( 'Convolution_3_weights' ):
        W_conv3 = weight_variable([3, 3, 64, 64]) # idem for the third convolution layer
    with tf.name_scope( 'Convolution_3_bias' ):
        b_conv3 = bias_variable([64])
    
    with tf.name_scope( 'Fully_connected_ReLU_weights' ):
        W_fc1 = weight_variable([1600, 512]) # idem for the fully connected layer
    with tf.name_scope( 'Fully_connected_ReLU_bias' ):
        b_fc1 = bias_variable([512])

    with tf.name_scope( 'Final_matrix_multiplier_weights' ):
        W_fc2 = weight_variable([512, ACTIONS]) # idem for the last layer ; the shape of the output is the  shape of the input vector to the emulator
    with tf.name_scope( 'Final_matrix_multiplier_bias' ):
        b_fc2 = bias_variable([ACTIONS])

    # input layer
    with tf.name_scope( 'Frame_input' ):
        s = tf.placeholder("float", [None, 80, 80, 4]) # placeholder defines the type and shape the input. Will be evaluated in session

    # couches internes
    with tf.name_scope( 'Convolution_1' ):
        h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1) # ReLU : Rectified Linear Unit / f(x) = max(0,x) activation function. Introduce non-linearity
    #with tf.name_scope( Maxpool_2 ):
    #    h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope( 'Convolution_2' ):
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2) # ReLU

    with tf.name_scope( 'Convolution_3' ):
        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 2) + b_conv3) # ReLU

    with tf.name_scope( 'Reshape' ):
        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600]) # set h_conv3 from a dim4 tensor to a XX*1600 matrix

    with tf.name_scope( 'Fully_connected_ReLU' ):
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1) # fully-connected layer with the weights W_fc1, bias b_fc1, and activation function ReLU

    # readout layer
    with tf.name_scope( 'Final_matrix_multiplier' ):
        readout = tf.matmul(h_fc1, W_fc2) + b_fc2 # last neural layer with no activation function returning a vector of size ACTIONS

    return s, readout, h_fc1 # s is the input, readout the output vector, h_fc1 the value of the the layer before the output

def trainNetwork(s, readout, h_fc1, sess):
    # defining loss function
    print("Debut de l'entrainement")
    with tf.name_scope( 'Action_inputs' ):
        a = tf.placeholder("float", [None, ACTIONS])
    with tf.name_scope( 'Target_Q_TD_estimation' ):
        y = tf.placeholder("float", [None]) # placeholder will be evaluated in session
    x = tf.mul(readout,a)
    with tf.name_scope( 'Current_Q_estimation' ):
        readout_action = tf.reduce_sum(tf.mul(readout, a), reduction_indices = 1) # tf.mul returns the product element-wise of readout and a. tf.reduce_sum sums these terms and returns a tensor of shape dim(a)-1
    with tf.name_scope( 'Loss_function' ):
        cost = tf.reduce_mean(tf.square(tf.sub(y, readout_action))) # returns the mean of (y_i - r_i)^2
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost) # applies the Adam algorithm to minimise the loss function using gradient. 1e-6 is the learning rate
    
    score = 0
    score_final=score
    partie = 1
    # for TensorBoard
    #tf.summary.scalar("loss_function",cost)
    #tf.summary.histogram("loss_histogram",cost)
    #tf.summary.histogram("readout",readout)
    #tf.summary.histogram("TD_target",y)
    #summary_op = tf.summary.merge_all()
 
    # record of the last state transitions
    D = deque()
    
    # saving
    a_file = open("logs_atari/readout.txt",  'w' )
    score_file = open("logs_atari/score.txt",  'w' )

    # Get the first state with no action and convert the frame to greyscale 80*80
    x_t, r_0, terminal = ale.getScreenGrayscale(), ale.act(0), ale.game_over()
    x_t = cv2.resize(x_t, (80,80))
    s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2) # concatenate x_t vectors on dimension 2. s_t is a dim 4 tensor representing the last 4 consecutives frames, ie the actual state

    # saving and loading networks
    saver = tf.train.Saver() # save variables (weights and bias)
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("logs_atari") # get a checkpoint on the directory "logs_atari"
    if checkpoint and checkpoint.model_checkpoint_path: # model_checkpoint_path is the last saved network. This checks a network has been previously created and saved
        saver.restore(sess, checkpoint.model_checkpoint_path) # restore variables from checkpoint.model_checkpoint_path for the ongoing session
        print "Network succesfully loaded :", checkpoint.model_checkpoint_path
    else:
        print "No network found. Creating a new network randomly initialized."

    epsilon = INITIAL_EPSILON
    t = OFFSET
    
    # file_writer = tf.summary.FileWriter("logs_atari", sess.graph) # save the graph on logs_atari (for TensorBoard)
    while "pigs" != "fly":
        # select a random action with probability epsilon
        readout_t = readout.eval(feed_dict = {s : [s_t]})[0] # evaluate index 0 of readout for s = [s_t]
        a_t = np.zeros([ACTIONS]) # a_t initialized as zeros
        action_index = 0
        if random.random() <= epsilon or t <= OBSERVE + OFFSET: # select random action
            action_index = random.randrange(ACTIONS)
            action = legal_actions[action_index] # if legal_actions is size 16, action=action_index. If the legal actions in the game are fewer than that, action != action_index
        else:
            action_index = np.argmax(readout_t) # select the action that maximize readout
            action = legal_actions[action_index]
        # send action to the emulator, a_t is used by the network for the loss function
        a_t[action_index]=1
        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE + OFFSET: # reduce epsilon (shifting from random to the learned policy)
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        for i in range(0, K): # to apply the chosen action during K frames
            # apply the action and observe the following state and reward
            r_t, terminal = ale.act(action), ale.game_over() # get the score and game_over boolean
            score += r_t
            x_t1 = ale.getScreenGrayscale() # actual frame for new state
            x_t1 = cv2.resize(x_t1, (80, 80))
            x_t1 = np.reshape(x_t1, (80, 80, 1)) # reshaping the tensor
            s_t1 = np.append(x_t1, s_t[:,:,0:3], axis = 2) # s_t1 (ie s_t+1) is a copy of x_t1 concatenated with s_t[:,:,0:3] on axis 2
            # the actual is then the last 4 frames
            if terminal : # ie game over
                ale.reset_game()
                score_final=score
                score_file.write(str(partie) + "," + str(score) +  '\n' )
                partie +=1
                score = 0
                print ("Score pour cette partie : ", score_final)
            # save the state transition
            D.append((s_t, a_t, r_t, s_t1, terminal))
            if len(D) > REPLAY_MEMORY:
                D.popleft()

        # learn if done observing
        if t > OBSERVE + OFFSET :
            #
            minibatch = random.sample(D, BATCH) # select a sample a size BATCH amongst D

            # extracting variables from the sample
            s_j_batch = [d[0] for d in minibatch] # initial states
            a_batch = [d[1] for d in minibatch]  # actions
            r_batch = [d[2] for d in minibatch] # rewards
            s_j1_batch = [d[3] for d in minibatch]  # final states
            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch}) # readout for the input = s_j1_batch
            for i in range(0, len(minibatch)):
                if minibatch[i][4]: # terminal = True ie game over
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i])) # readout_j1_batch[i] is the vector of reward for each input from the sample
                    # select using np.max the estimated maximum reward

            # apply gradient descent
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch})
                
            #file_writer.add_summary(summary, t) # for Tensorboard
            
        # update parameters
        s_t = s_t1
        t += 1

        # save every 1000 iteration in logs_atari
        if t % 1000 == 0:
            saver.save(sess,  'logs_atari/atari-dqn' , global_step = t)
        

        a_file.write(str(t) +"  "+",".join([str(x) for x in readout_t]) +  '\n' )
        #cv2.imwrite("logs_atari/frame" + str(t) + ".png", x_t1) # to get screenshots

            
def playGame():
    print "Creating session..."
    sess = tf.InteractiveSession()
    print("Session created. Creating network ...")
    s, readout, h_fc1 = createNetwork()
    print("Networks created. Learning started.")       
    trainNetwork(s, readout, h_fc1, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
