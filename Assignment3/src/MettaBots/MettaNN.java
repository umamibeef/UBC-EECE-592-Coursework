package MettaBots;

import NeuralNetMulti.*;

import robocode.*;

import java.awt.*;
import java.awt.event.KeyEvent;
import java.io.*;
import java.util.ArrayList;
import java.util.Random;

/**
 * In order to allow this particular robot to work, robocode.robot.filesystem.quota=4294967296
 * was set in robocode.properties in order to allow large files to be loaded and saved.
 */

@SuppressWarnings("Duplicates")
public class MettaNN extends AdvancedRobot //Robot
{
    // Learning constants
    private static final int NO_LEARNING_RANDOM = 0; // No learning, completely random, baselines behaviour
    private static final int NO_LEARNING_GREEDY = 1; // No learning, will pick greediest move if LUT is available
    private static final int SARSA = 2; // On-policy SARSA
    private static final int Q_LEARNING = 3; // Off-policy Q-learning
    private static final boolean NON_TERMINAL_STATE = false;
    private static final boolean TERMINAL_STATE = true;

    // Neural network parameters
    private static final int NUM_INPUTS = 5;            // Number of NN inputs
    private static final int NUM_HIDDEN_NEURONS = 200;  // Number of NN hidden neurons
    private static final int NUM_OUTPUTS = 8;           // Number of NN outputs
    private static final int MIN_VAL = -1;              // Minimum value for activation function (sigmoid)
    private static final int MAX_VAL = 1;               // Maximum value for activation function (sigmoid)
    private static final double MOMENTUM = 0.2;         // Momentum parameter for backpropagation
    private static final double LEARNING_RATE = 0.005;  // Learning rate parameter for backpropagation
    private static final double WEIGHT_INIT_MIN = -1.0; // Random weight init low limit
    private static final double WEIGHT_INIT_MAX = 1.0;  // Random weight init high limit

    // Reinforcement learning parameters
    private static final double ALPHA = 0.5;    // Fraction of difference used
    private static final double GAMMA = 0.8;    // Discount factor
    private static final double EPSILON = 0.1;  // Probability of exploration
    //private static final double EPSILON = 1.0;  // Probability of exploration
    //private int mCurrentLearningPolicy = NO_LEARNING_RANDOM;
    //private int mCurrentLearningPolicy = NO_LEARNING_GREEDY;
    private int mCurrentLearningPolicy = SARSA;
    //private int mCurrentLearningPolicy = Q_LEARNING;
    private boolean mIntermediateRewards = true;
    private boolean mTerminalRewards = true;
    private static final int REWARD_SCALER = 275; // How much to scale the rewards by for the neural network calculation

    // Debug
    private static boolean mDebug = true;

    // Misc. constants used in the robot
    private static final int ARENA_SIZEX_PX = 800;
    private static final int ARENA_SIZEY_PX = 600;
    private static final int ENEMY_ENERGY_THRESHOLD = 50;
    private static final int NULL_32 = 0xFFFFFFFF;

    // Move actions
    private static final int ACTION_MOVE_UP = 0;
    private static final int ACTION_MOVE_DN = 1;
    private static final int ACTION_MOVE_LT = 2;
    private static final int ACTION_MOVE_RT = 3;
    private static final int ACTION_MOVE_NUM = 4;
    private static final int ACTION_MOVE_DISTANCE = 50;
    // Fire actions
    private static final int ACTION_FIRE_0 = 0;
    private static final int ACTION_FIRE_3 = 1;
    private static final int ACTION_FIRE_NUM = 2;
    // Action indices
    private static final int ACTION_INDEX_UP_NO_FIRE = ACTION_MOVE_UP;
    private static final int ACTION_INDEX_DN_NO_FIRE = ACTION_MOVE_DN;
    private static final int ACTION_INDEX_LT_NO_FIRE = ACTION_MOVE_LT;
    private static final int ACTION_INDEX_RT_NO_FIRE = ACTION_MOVE_RT;
    private static final int ACTION_INDEX_UP_FIRE = ACTION_MOVE_UP + ACTION_MOVE_NUM;
    private static final int ACTION_INDEX_DN_FIRE = ACTION_MOVE_DN + ACTION_MOVE_NUM;
    private static final int ACTION_INDEX_LT_FIRE = ACTION_MOVE_LT + ACTION_MOVE_NUM;
    private static final int ACTION_INDEX_RT_FIRE = ACTION_MOVE_RT + ACTION_MOVE_NUM;
    // Action constants
    private static final int ACTION_DIMENSIONALITY = ACTION_MOVE_NUM * ACTION_FIRE_NUM;
    private static final int ACTION_MODE_MAX_Q = 0;
    private static final int ACTION_MODE_EPSILON_GREEDY = 1;

    // State constants
    private static final int STATE_POS_X_MAX = ARENA_SIZEX_PX;
    private static final int STATE_POS_Y_MAX = ARENA_SIZEY_PX;
    private static final int STATE_DISTANCE_MAX = 1000;
    private static final int STATE_DIMENSIONALITY = 5; // Total number of state dimensions
    private static final int STATE_POS_X_INDEX = 0;
    private static final int STATE_POS_Y_INDEX = 1;
    private static final int STATE_DISTANCE_INDEX = 2;
    private static final int STATE_HEADING_0_INDEX = 3;
    private static final int STATE_HEADING_1_INDEX = 4;

    // Neural network file and properties
    private static final String NN_WEIGHTS_FILE_NAME = "./Ass3NeuralNetWeights.dat";
    private File mNeuralNetWeightsFile;
    // Stat file and properties
    private static final String STATS_FILE_NAME = "./Ass3Stats.csv";
    private File mStatsFile;

    // Variables to track the state of the arena
    private double mRobotX;
    private double mRobotY;
    private double mRobotHeading;
    private double mRobotGunHeading;
    private double mRobotGunBearing;
    private double mRobotEnergy;
    private double mEnemyDistance;
    private double mEnemyHeading;
    private double mEnemyBearing;
    private double mEnemyBearingFromGun;
    private double mEnemyEnergy;
    private double mEnemyX;
    private double mEnemyY;

    private int mCurrentAction;
    private int mPreviousAction;
    private double [] mCurrentStateSnapshot = new double[STATE_DIMENSIONALITY];
    private double [] mPreviousStateSnapshot = new double[STATE_DIMENSIONALITY];

    private double mPreviousEnergyDifference;
    private double mCurrentEnergyDifference;
    private double mCurrentReward;

    // Neural network to approximate Q(s,a) function
    private static NeuralNetMulti mNeuralNet;
    private static ArrayList<ArrayList<ArrayList<Double>>> mNeuralNetWeights;

    // Completion conditions
    private final TurnCompleteCondition mTurnComplete = new TurnCompleteCondition(this);
    private final MoveCompleteCondition mMoveComplete = new MoveCompleteCondition(this);
    private final GunTurnCompleteCondition mGunMoveComplete = new GunTurnCompleteCondition(this);

    // Winrate tracking for every 100 rounds
    private static final int NUM_ROUNDS = 500000;
    private static final int NUM_ROUNDS_DIV_100 = NUM_ROUNDS / 100;
    private static int [] mNumWinArray = new int[NUM_ROUNDS_DIV_100];
    private static double [] mAverageDeltaQ = new double[500000];
    private static double mRoundTotalDeltaQ;
    private static int mRoundDeltaQNum = 1;

    public void run()
    {
        int actionIndex;
        long fileSize;
        ArrayList<ArrayList<ArrayList<Double>>> neuralNetworkWeights;

        // Instantiate a new neural network for the robot to learn
        mNeuralNet = new NeuralNetMulti(
            NUM_INPUTS, NUM_OUTPUTS, NUM_HIDDEN_NEURONS, LEARNING_RATE, MOMENTUM, MIN_VAL, MAX_VAL, WEIGHT_INIT_MIN, WEIGHT_INIT_MAX);

        // Set robot colours
        setColors(Color.PINK, Color.PINK, Color.PINK, Color.PINK, Color.PINK);
        // Set robot properties
        setAdjustGunForRobotTurn(true);
        setAdjustRadarForGunTurn(true);
        setAdjustRadarForRobotTurn(true);

        // Ask robocode for the neural network weights file
        mNeuralNetWeightsFile = getDataFile(NN_WEIGHTS_FILE_NAME);
        // Ask robocode for the stats file
        mStatsFile = getDataFile(STATS_FILE_NAME);

        // Get the neural network weights file size
        fileSize = mNeuralNetWeightsFile.length();
        printDebug("Current neural network file size %d\n", fileSize);
        // If the current file is empty, put in an empty hashmap
        if (fileSize == 0)
        {
            printDebug("No weights, initializing new ones...\n");
            newWeightFile(mNeuralNetWeightsFile);
            // Initialize weights for a new training session
            mNeuralNet.initializeWeights();
            mNeuralNetWeights = mNeuralNet.getWeights();
            saveWeights(mNeuralNetWeightsFile);
        }
        else
        {
            printDebug("Found weights, loading them...\n");
            // Load the weights from the file
            loadWeights(mNeuralNetWeightsFile);
            mNeuralNet.setWeights(mNeuralNetWeights);
        }

        printDebug("Data available: %d bytes\n", getDataQuotaAvailable());

        // If SARSA, we must take an action at start
        // Choose an action hash that has the maximum Q for this state
        if(mCurrentLearningPolicy == SARSA)
        {
            // Take a snapshot of the current state
            mCurrentStateSnapshot = takeStateSnapshot();
            actionIndex = getAction(ACTION_MODE_EPSILON_GREEDY, mCurrentStateSnapshot);
            // Take an action based on the current state
            takeAction(actionIndex);
            // Record our previous state snapshot
            mPreviousStateSnapshot = mCurrentStateSnapshot;
        }

        // Robot's infinite loop
        for (;;)
        {
            turnRadarRight(20);
        }
    }

    // Called when a key has been pressed
    public void onKeyPressed(java.awt.event.KeyEvent e)
    {
        switch (e.getKeyCode())
        {
            // Nothing here for now
            default:
                break;
        }
    }

    // Called when a key has been released (after being pressed)
    public void onKeyReleased(KeyEvent e)
    {
        switch (e.getKeyCode())
        {
            // Nothing here for now
            default:
                break;
        }
    }

    public void onScannedRobot(ScannedRobotEvent event)
    {
        double angle;

        printDebug("==[SCAN]==========================================\n");
        // Obtain state information
        // Robot's info
        mRobotX = getX();
        mRobotY = getY();
        mRobotHeading = getHeading();
        mRobotGunHeading = getGunHeading();
        mRobotGunBearing = normalizeAngle(mRobotHeading - mRobotGunHeading);
        mRobotEnergy = getEnergy();

        // Enemy's info
        mEnemyDistance = event.getDistance();
        mEnemyHeading = event.getHeading();
        mEnemyBearing = event.getBearing();
        mEnemyBearingFromGun = normalizeAngle(mRobotGunBearing + mEnemyBearing);
        mEnemyEnergy = event.getEnergy();
        // Calculate the enemy's last know position
        // Calculate the angle to the scanned robot
        angle = Math.toRadians(getHeading() + event.getBearing() % 360);
        // Calculate the coordinates of the robot
        mEnemyX = (getX() + Math.sin(angle) * event.getDistance());
        mEnemyY = (getY() + Math.cos(angle) * event.getDistance());

        printDebug("Robot: X %1.3f Y %1.3f Heading %1.3f GunHeading %1.3f Energy %1.3f\n",
        mRobotX, mRobotY, mRobotHeading, mRobotGunHeading, mRobotEnergy);
        printDebug("Enemy: Distance %1.3f Heading %1.3f Bearing %1.3f BearingFromGun %1.3f Energy %1.3f\n",
        mEnemyDistance, mEnemyHeading, mEnemyBearing, mEnemyBearingFromGun, mEnemyEnergy);

        learn(NON_TERMINAL_STATE);
    }

    private void learn(boolean terminalState)
    {
        double qPrevNew, qPrevOld, qNext;

        double [] currentActionQs;
        double [] previousActionQs;
        double [] previousActionQsUpdated;
        int actionIndex, index;

        printDebug("==[LEARN]=========================================\n");

        // Take a snapshot of the current state
        mCurrentStateSnapshot = takeStateSnapshot();
        // Feed forward the current state to the neural network
        currentActionQs = mNeuralNet.outputFor(mCurrentStateSnapshot);
        // Feed forward the previous state to the neural network
        previousActionQs = mNeuralNet.outputFor(mPreviousStateSnapshot);

        // Calculate the current reward
        // Reward can be obtained asynchronously through events such as bullet hitting
        // Add to the current reward the CHANGE in difference between previous and current
        if (mIntermediateRewards)
        {
            mPreviousEnergyDifference = mCurrentEnergyDifference;
            mCurrentEnergyDifference = mRobotEnergy - mEnemyEnergy;
            mCurrentReward += mCurrentEnergyDifference - mPreviousEnergyDifference;
            mCurrentReward /= REWARD_SCALER;
        }
        printDebug("Current reward (unscaled): %f\n", mCurrentReward * REWARD_SCALER);
        printDebug("Current reward (scaled): %f\n", mCurrentReward);

        switch (mCurrentLearningPolicy)
        {
            // No learning at all (baseline)
            case NO_LEARNING_RANDOM:
                printDebug("No learning and random!\n");
                // Take random action
                actionIndex = getRandomInt(0, ACTION_DIMENSIONALITY - 1);
                printDebug("Taking random action of 0x%02x\n", actionIndex);
                takeAction(actionIndex);
                break;
            case NO_LEARNING_GREEDY:
                printDebug("No learning and totally greedy!\n");
                // Get the action hash that has the maximum Q for this state
                printDebug("Find the maximum Q action for this new state");
                actionIndex = getAction(ACTION_MODE_MAX_Q, mCurrentStateSnapshot);
                // Take the action
                takeAction(actionIndex);
                break;
            // On-policy (SARSA)
            case SARSA:
                printDebug("SARSA!\n");
                // Choose an on-policy action
                actionIndex = getAction(ACTION_MODE_EPSILON_GREEDY, mCurrentStateSnapshot);

                // DEBUG
                //printDebug("Output before training:\n");
                //for (index = 0; index < NUM_OUTPUTS; index++)
                //{
                //    printDebug("[%d: % .16f]\n", index, previousActionQs[index]);
                //}

                // Calculate new value for previous Q;
                qNext = currentActionQs[actionIndex];
                qPrevOld = previousActionQs[mPreviousAction];
                qPrevNew = calculateQPrevNew(qNext, qPrevOld);

                mRoundTotalDeltaQ += Math.abs(qPrevNew - qPrevOld);
                mRoundDeltaQNum++;

                // DEBUG
                printDebug("Replacing index %d % .16f -> % .16f\n", mPreviousAction, qPrevOld, qPrevNew);
                printDebug("Total backpropagations this round: %d\n", mRoundDeltaQNum);
                printDebug("Delta is %f, round average %f\n", Math.abs(qPrevNew - qPrevOld), mRoundTotalDeltaQ/mRoundDeltaQNum);

                // Backpropagate the action through the neural network
                // Replace the old previous Q value with the new one
                previousActionQs[mPreviousAction] = qPrevNew;
                // Train the neural network with the new dataset
                mNeuralNet.train(createTrainingSet(mPreviousStateSnapshot, previousActionQs));

                // DEBUG
                //previousActionQsUpdated = mNeuralNet.outputFor(mPreviousStateSnapshot);
                //printDebug("Output after training:\n");
                //for (index = 0; index < NUM_OUTPUTS; index++)
                //{
                //    printDebug("[%d: % .16f]\n", index, previousActionQsUpdated[index]);
                //}

                // Reset reward until the next learn
                mCurrentReward = 0.0;
                if (terminalState)
                {
                    // We're done! No more actions.
                    printDebug("Terminal state! No more actions available.\n");
                    return;
                }
                // Take the next action
                takeAction(actionIndex);
                break;
            // Off-policy (Q-Learning)
            case Q_LEARNING:
                printDebug("Q-learning!\n");
                if (terminalState)
                {
                    // We're done! No more actions.
                    printDebug("Terminal state! No more actions available.\n");
                    // Learn from the terminal action
                    // Calculate new value for previous Q, next Q is zero since we are terminal
                    qPrevOld = previousActionQs[mPreviousAction];
                    qPrevNew = calculateQPrevNew(0.0, qPrevOld);
                    // Backpropagate the action through the neural network
                    // Replace the old previous Q value with the new one
                    previousActionQs[mPreviousAction] = qPrevNew;
                    // Train the neural network with the new dataset
                    mNeuralNet.train(createTrainingSet(mPreviousStateSnapshot, previousActionQs));
                    return;
                }
                else
                {
                    // Choose an on-policy action
                    printDebug("Choosing on-policy action to take");
                    actionIndex = getAction(ACTION_MODE_EPSILON_GREEDY, mCurrentStateSnapshot);
                    // Take the action
                    takeAction(actionIndex);
                    // Record our previous state snapshot
                    mPreviousStateSnapshot = mCurrentStateSnapshot;
                    // Observe the new environment
                    mCurrentStateSnapshot = takeStateSnapshot();
                    // Feed forward the current state to the neural network
                    currentActionQs = mNeuralNet.outputFor(mCurrentStateSnapshot);
                    // Get the action hash that has the maximum Q for this state
                    printDebug("Find the maximum Q action for this new state");
                    actionIndex = getAction(ACTION_MODE_MAX_Q, mCurrentStateSnapshot);
                    // Calculate new value for previous Q;
                    qNext = currentActionQs[actionIndex];
                    qPrevOld = previousActionQs[mPreviousAction];
                    qPrevNew = calculateQPrevNew(qNext, qPrevOld);
                    // Backpropagate the action through the neural network
                    // Replace the old previous Q value with the new one
                    previousActionQs[mPreviousAction] = qPrevNew;
                    // Train the neural network with the new dataset
                    mNeuralNet.train(createTrainingSet(mPreviousStateSnapshot, previousActionQs));

                    // Reset reward until the next learn
                    mCurrentReward = 0.0;
                }
                break;
            default:
                break;
        }

        // Record our previous state snapshot
        mPreviousStateSnapshot = mCurrentStateSnapshot;
    }

    /**
     * This function will either explore randomly or take an 1-EPSILON greedy action that is passed into it
     * @param actionIndex The action to take
     */
    private void takeAction(int actionIndex)
    {
        int moveDirection = ACTION_MOVE_UP;
        int fireType = ACTION_FIRE_3;
        double angle, newHeading, estimatedEnemyBearingFromGun;

        // Record our current action
        mCurrentAction = actionIndex;

        switch (actionIndex)
        {
            case ACTION_INDEX_UP_NO_FIRE:
                moveDirection = ACTION_MOVE_UP;
                fireType = ACTION_FIRE_0;
                break;
            case ACTION_INDEX_DN_NO_FIRE:
                moveDirection = ACTION_MOVE_DN;
                fireType = ACTION_FIRE_0;
                break;
            case ACTION_INDEX_LT_NO_FIRE:
                moveDirection = ACTION_MOVE_LT;
                fireType = ACTION_FIRE_0;
                break;
            case ACTION_INDEX_RT_NO_FIRE:
                moveDirection = ACTION_MOVE_RT;
                fireType = ACTION_FIRE_0;
                break;
            case ACTION_INDEX_UP_FIRE:
                moveDirection = ACTION_MOVE_UP;
                fireType = ACTION_FIRE_3;
                break;
            case ACTION_INDEX_DN_FIRE:
                moveDirection = ACTION_MOVE_DN;
                fireType = ACTION_FIRE_3;
                break;
            case ACTION_INDEX_LT_FIRE:
                moveDirection = ACTION_MOVE_LT;
                fireType = ACTION_FIRE_3;
                break;
            case ACTION_INDEX_RT_FIRE:
                moveDirection = ACTION_MOVE_RT;
                fireType = ACTION_FIRE_0;
                break;
            default:
                break;
        }

        // Perform the move action
        newHeading = -1 * normalizeAngle(getHeading());
        switch (moveDirection)
        {
            case ACTION_MOVE_UP:
                break;
            case ACTION_MOVE_DN:
                newHeading = normalizeAngle(newHeading + 180);
                break;
            case ACTION_MOVE_LT:
                newHeading = normalizeAngle(newHeading + 270);
                break;
            case ACTION_MOVE_RT:
                newHeading = normalizeAngle(newHeading + 90);
                break;
            default:
                // We should never be in here, do nothing.
                break;
        }
        setTurnRight(newHeading);

        // Execute the turn
        execute();
        waitFor(mTurnComplete);
        setAhead(ACTION_MOVE_DISTANCE);
        // Execute the ahead
        execute();
        waitFor(mMoveComplete);

        // Re-calculate the enemy's bearing based on its last know position
        // Calculate gun turn to predicted x,y location
        angle = absoluteBearing(getX(), getY(), mEnemyX, mEnemyY);
        // Turn the gun to the predicted x,y location
        estimatedEnemyBearingFromGun = normalizeAngle((int) (angle - getGunHeading()));

        // Perform the aim type action
        // Aim straight now
        turnGunRight(estimatedEnemyBearingFromGun);

        // Execute the aim right away
        execute();
        waitFor(mGunMoveComplete);

        // Perform the firing type action
        switch (fireType)
        {
            case ACTION_FIRE_0:
                // We don't fire in this case
                break;
            case ACTION_FIRE_3:
                // Fire a 3 power bullet
                setFireBullet(3.0);
                break;
            default:
                // We should never be in here, do nothing.
                break;
        }
        // Execute the fire action
        execute();

        // Update the previous action index
        mPreviousAction = mCurrentAction;
    }

    /**
     * Creates an ArrayList training set suitable for training the neural network
     * @param inputVectorArray
     * @param outputVectorArray
     * @return
     */
    private ArrayList<ArrayList<Double>> createTrainingSet(double [] inputVectorArray, double [] outputVectorArray)
    {
        int i;
        ArrayList<ArrayList<Double>> trainingSet = new ArrayList<>();
        ArrayList<Double> inputVector = new ArrayList<>();
        ArrayList<Double> outputVector = new ArrayList<>();

        // Convert ArrayLists into static arrays
        for(i = 0; i < inputVectorArray.length; i++)
        {
            inputVector.add(inputVectorArray[i]);
        }
        for(i = 0; i < outputVectorArray.length; i++)
        {
            outputVector.add(outputVectorArray[i]);
        }

        trainingSet.add(inputVector);
        trainingSet.add(outputVector);

        return trainingSet;
    }


    /**
     * Updates the previous Q value based on the passed in next Q value and the learning hyperparameters
     * @param qNext The next Q value
     * @param qPrevOld The old previous Q value to be updated
     * @return
     */
    private double calculateQPrevNew(double qNext, double qPrevOld)
    {
        double qPrevNew;

        qPrevNew = qPrevOld + (ALPHA * (mCurrentReward + (GAMMA * qNext) - qPrevOld));

        return qPrevNew;
    }

    /**
     * Obtain an action based on the mode passed in
     *
     * @param mode The action selection mode, either epsilon greedy or qmax
     * @param currentStateSnapshot Get the action with the maximum Q from the provided state hash
     * @return An action based on the mode passed in
     */
    private int getAction(int mode, double [] currentStateSnapshot)
    {
        int index, selectedAction;
        ArrayList<Integer> qMaxActions = new ArrayList<>();
        double [] actionQs;
        double qVal, randomDouble;
        double qMax = -999.0;

        // Feed forward current state snapshot into neural network and obtain a set of action Q values
        actionQs = mNeuralNet.outputFor(currentStateSnapshot);

        // Get the maximum action
        for (index = 0; index < ACTION_DIMENSIONALITY; index++)
        {
            qVal = actionQs[index];

            // Update current max
            if (qVal > qMax)
            {
                // New max, clear array
                // We can have a maximum of the number of possible actions as the number of possible actions
                qMaxActions = new ArrayList<>();
                qMaxActions.add(index);
                qMax = qVal;
            }
            else if (qVal == qMax)
            {
                // We found a q value equal to the max, add it to the possible actions
                qMaxActions.add(index);
            }
        }

        // Iterate through all possible actions
        printDebug("State Input: [% f, % f, % f, % f, % f]\n",
            currentStateSnapshot[0], currentStateSnapshot[1], currentStateSnapshot[2], currentStateSnapshot[3], currentStateSnapshot[4]);
        printDebug("Possible Q-values:\n");
        for (index = 0; index < ACTION_DIMENSIONALITY; index++)
        {
            printDebug("[%d]: % 1.10f\n", index, actionQs[index]);
        }

        printDebug("Max actions: %d\n", qMaxActions.size());
        if (qMaxActions.size() == 1)
        {
            selectedAction = qMaxActions.get(0);
            printDebug("Found best possible action to take [%d:% f]\n",
            selectedAction, actionQs[selectedAction]);
        }
        else
        {
            selectedAction = getRandomInt(0, qMaxActions.size());
            printDebug("Found %d possible actions to take, randomly picking action [%d:% f]\n",
            actionQs.length, selectedAction, actionQs[selectedAction]);
        }


        switch (mode)
        {
            // If we're choosing epsilon greedy, then we must choose between max Q or exploratory, so do that here
            case ACTION_MODE_EPSILON_GREEDY:
                // Roll the dice
                printDebug("Greedy chance of %f, exploring chance of %f\n", (1 - EPSILON), EPSILON);
                randomDouble = getRandomDouble(0.0, 1.0);
                if (randomDouble < EPSILON)
                {
                    // Take random action
                    selectedAction = getRandomInt(0, ACTION_DIMENSIONALITY - 1);
                    printDebug("Got random number %1.3f - picking random action of [%d:% f]\n",randomDouble, selectedAction, actionQs[selectedAction]);
                }
                else
                {
                    // Take greedy action
                    printDebug("Picking greedy action of [%d:% f]\n", selectedAction, actionQs[selectedAction]);
                }
                break;
            // We should already have max Q from above, so choose that
            case ACTION_MODE_MAX_Q:
                printDebug("Picking max Q action of [%d:% f]\n", selectedAction, actionQs[selectedAction]);
                break;
            default:
                // We should never be here
                break;
        }

        return selectedAction;
    }

    public void onBulletHit(BulletHitEvent event)
    {
        if (mIntermediateRewards)
        {
            mCurrentReward += 30.0;
        }
    }

    public void onHitByBullet(HitByBulletEvent event)
    {
        if (mIntermediateRewards)
        {
            mCurrentReward -= 30.0;
        }
    }

    public void onBattleEnded(BattleEndedEvent event)
    {
        // Save the NN weights to the data file
        mNeuralNetWeights = mNeuralNet.getWeights();
        saveWeights(mNeuralNetWeightsFile);
        // Save the win tracker to the tracker file
        saveStats(mStatsFile);
    }

    public void endOfRoundStats()
    {
        mAverageDeltaQ[getRoundNum()] = mRoundTotalDeltaQ / mRoundDeltaQNum;

        printDebug("Round %d BATTLE ENDED! Current win rate %d\n", getRoundNum(), mNumWinArray[(getRoundNum() - 1) / 100]);
        printDebug("              Average delta Q % f from %d backpropagations\n", mAverageDeltaQ[getRoundNum()], mRoundDeltaQNum);

        mRoundTotalDeltaQ = 0.0;
        mRoundDeltaQNum = 1;
    }

    public void onDeath(DeathEvent event)
    {
        // Give terminal reward of -100
        if (mTerminalRewards)
        {
            mCurrentReward -= 100;
            learn(TERMINAL_STATE);
        }

        printDebug("Metta died.\n");
        endOfRoundStats();
    }

    public void onWin(WinEvent event)
    {
        // Record our number of wins for every 100 rounds
        mNumWinArray[(getRoundNum() - 1) / 100]++;

        // Give terminal reward of 100
        if (mTerminalRewards)
        {
            mCurrentReward += 100;
            learn(TERMINAL_STATE);
        }

        printDebug("Metta won.\n");
        endOfRoundStats();
    }

    /**
     * Normalize an angle
     *
     * @param angle The angle to normalize
     * @return The normalized angle
     */
    private double normalizeAngle(double angle)
    {
        double result = angle;

        while (result > 180) result -= 360;
        while (result < -180) result += 360;

        return result;
    }

    private double[] getQValues(ArrayList<Double> preprocessedState)
    {
        int i;
        double[] state = new double[preprocessedState.size()];

        // Neural network takes in an array, so we have to convert the ArrayList
        for (i = 0; i < preprocessedState.size(); i++)
        {
            state[i] = preprocessedState.get(i);
        }

        return mNeuralNet.outputFor(state);
    }

    /**
     * Generate a preprocessed state snapshot and save it in the related variables
     * @return The state snapshot array that can be take
     */
    private double [] takeStateSnapshot()
    {
        double [] stateSnapshot = new double[STATE_DIMENSIONALITY];

        // Scale state values and save them to the current stat snapshot
        stateSnapshot[STATE_POS_X_INDEX] = scaleValue(mRobotX, 0, STATE_POS_X_MAX, MIN_VAL, MAX_VAL);
        stateSnapshot[STATE_POS_Y_INDEX] = scaleValue(mRobotY, 0, STATE_POS_Y_MAX, MIN_VAL, MAX_VAL);
        stateSnapshot[STATE_DISTANCE_INDEX] = scaleValue(mEnemyDistance, 0, STATE_DISTANCE_MAX, MIN_VAL, MAX_VAL);

        if (mRobotHeading >= 0 && mRobotHeading < 90)
        {
            stateSnapshot[STATE_HEADING_0_INDEX] = -1.0;
            stateSnapshot[STATE_HEADING_1_INDEX] = -1.0;
        }
        else if (mRobotHeading >= 90 && mRobotHeading < 180)
        {
            stateSnapshot[STATE_HEADING_0_INDEX] = 1.0;
            stateSnapshot[STATE_HEADING_1_INDEX] = -1.0;
        }
        else if (mRobotHeading >= 180 && mRobotHeading < 270)
        {
            stateSnapshot[STATE_HEADING_0_INDEX] = -1.0;
            stateSnapshot[STATE_HEADING_1_INDEX] = 1.0;
        }
        else if (mRobotHeading >= 270 && mRobotHeading < 360)
        {
            stateSnapshot[STATE_HEADING_0_INDEX] = 1.0;
            stateSnapshot[STATE_HEADING_1_INDEX] = 1.0;
        }

        //printDebug("Preprocessed state values: %1.3f %1.3f %1.3f %1.3f %1.3f\n",
        //        stateSnapshot[STATE_POS_X_INDEX],
        //        stateSnapshot[STATE_POS_Y_INDEX],
        //        stateSnapshot[STATE_DISTANCE_INDEX],
        //        stateSnapshot[STATE_HEADING_0_INDEX],
        //        stateSnapshot[STATE_HEADING_1_INDEX]);

        return stateSnapshot;
    }

    /**
     * Returns an absolute bearing between two points
     *
     * @param x0 Point 0's x coordinate
     * @param y0 Point 0's y coordinate
     * @param x1 Point 1's x coordinate
     * @param y1 Point 1's y coordinate
     * @return Returns an absolute bearing based on the coordinates entered
     */
    private double absoluteBearing(double x0, double y0, double x1, double y1)
    {
        double xo = x1 - x0;
        double yo = y1 - y0;
        double hyp = calculateDistance(x0, y0, x1, y1);
        double asin = Math.toDegrees(Math.asin(xo / hyp));
        double bearing = 0;

        if (xo > 0 && yo > 0)
        {
            // both pos: lower-Left
            bearing = asin;
        }
        else if (xo < 0 && yo > 0)
        {
            // x neg, y pos: lower-right
            bearing = 360 + asin; // arcsin is negative here, actually 360 - ang
        }
        else if (xo > 0 && yo < 0)
        {
            // x pos, y neg: upper-left
            bearing = 180 - asin;
        }
        else if (xo < 0 && yo < 0)
        {
            // both neg: upper-right
            bearing = 180 - asin; // arcsin is negative here, actually 180 + ang
        }

        return bearing;
    }

    /**
     * Returns the distance between two coordinates
     *
     * @param x0 X position of the first coordinate
     * @param y0 Y position of the first coordinate
     * @param x1 X position of the second coordinate
     * @param y1 Y position of the second coordinate
     * @return A double value representing the distance between two coordinates
     */
    private double calculateDistance(double x0, double y0, double x1, double y1)
    {
        double distance;

        distance = Math.sqrt(Math.pow((x1 - x0), 2) + Math.pow((y1 - y0), 2));

        return distance;
    }

    /**
     * Updates a field in an int
     *
     * @param inputInteger The input integer to modify
     * @param fieldWidth   The width of the field to modify
     * @param fieldOffset  The field's offset
     * @param value        The value to update into the field
     * @return The updated input integer
     */
    private int updateIntField(int inputInteger, int fieldWidth, int fieldOffset, int value)
    {
        int returnValue;
        int mask;

        returnValue = inputInteger;

        // Create mask
        mask = ~(((1 << fieldWidth) - 1) << fieldOffset);
        // Mask out field from input
        returnValue &= mask;
        // OR in the new value into the field
        returnValue |= value << fieldOffset;

        return returnValue;
    }

    /**
     * Returns the value of a field in an int
     *
     * @param inputInteger The input integer to extract the value from
     * @param fieldWidth   The width of the field to extract
     * @param fieldOffset  The offset of the field to extract
     * @return Returns the value in the selected field
     */
    private int getIntFieldVal(int inputInteger, int fieldWidth, int fieldOffset)
    {
        int returnValue;
        int mask;

        returnValue = inputInteger;

        // Create mask
        mask = ((1 << fieldWidth) - 1) << fieldOffset;
        // Mask out the field from the input
        returnValue &= mask;
        // Shift down to grab it
        returnValue >>>= fieldOffset;

        return returnValue;
    }

    /**
     * Returns a random double value between specified min and max values
     *
     * @param min minimum number random number can be
     * @param max maximum number random number can be
     * @return a random double between specified min and max
     */
    private double getRandomDouble(double min, double max)
    {
        double random, result;

        random = new Random().nextDouble();
        result = min + (random * (max - min));

        return result;
    }

    /**
     * Returns a random integer value between specified min and max values
     *
     * @param min minimum number random number can be inclusive
     * @param max maximum number random number can be inclusive
     * @return a random integer between specified min and max
     */
    private int getRandomInt(int min, int max)
    {
        int result;
        Random random;

        random = new Random();
        result = random.nextInt(max - min + 1) + min;

        return result;
    }

    /**
     * Scale a value 0-max into a double with a specified min and max value
     */
    private double scaleValue(double value, double fromMin, double fromMax, double toMin, double toMax)
    {
        double scaledValue;

        scaledValue = (value - fromMin) * (toMax - toMin) / (fromMax - fromMin) + toMin;

        return scaledValue;
    }

    /**
     * Conditionally prints a message if the debug flag is on
     *
     * @param format    The string to format
     * @param arguments The string format's variables
     */
    private void printDebug(String format, Object... arguments)
    {
        if (mDebug)
        {
            System.out.format(format, arguments);
        }
    }

    /**
     * Create a new neural network weight file
     *
     * @param weightFile The filename to use for the neural network weights
     */
    private void newWeightFile(File weightFile)
    {
        try
        {
            printDebug("Creating neural network weights file...\n");
            RobocodeFileOutputStream fileOut = new RobocodeFileOutputStream(weightFile);
            ObjectOutputStream out = new ObjectOutputStream(new BufferedOutputStream(fileOut));
            out.writeObject(new ArrayList<ArrayList<ArrayList<Double>>>());
            out.close();
            fileOut.close();
        }
        catch (IOException exception)
        {
            exception.printStackTrace();
        }
    }

    /**
     * Save the neural network weights
     *
     * @param weightFile The filename to use for the neural network weights
     */
    private void saveWeights(File weightFile)
    {
        try
        {
            printDebug("Saving neural network weights to file...\n");
            RobocodeFileOutputStream fileOut = new RobocodeFileOutputStream(weightFile);
            ObjectOutputStream out = new ObjectOutputStream(new BufferedOutputStream(fileOut));
            out.writeObject(mNeuralNetWeights);
            out.close();
            fileOut.close();
        }
        catch (IOException exception)
        {
            exception.printStackTrace();
        }
    }

    /**
     * Load neural network weights
     *
     * @param weightFile The filename to use for the neural network weights
     */
    private void loadWeights(File weightFile)
    {
        try
        {
            printDebug("Loading neural network weights from file...\n");
            FileInputStream fileIn = new FileInputStream(weightFile);
            ObjectInputStream in = new ObjectInputStream(new BufferedInputStream(fileIn));
            mNeuralNetWeights = (ArrayList<ArrayList<ArrayList<Double>>>) in.readObject();
            in.close();
            fileIn.close();
        }
        catch (IOException | ClassNotFoundException exception)
        {
            exception.printStackTrace();
        }
    }

    /**
     * Save the win tracking statistics file
     *
     * @param statsFile The filename to use for stats file
     */
    private void saveStats(File statsFile)
    {
        int i;

        try
        {
            printDebug("Saving stats to file...\n");
            RobocodeFileOutputStream fileOut = new RobocodeFileOutputStream(statsFile);
            PrintStream out = new PrintStream(new BufferedOutputStream(fileOut));
            out.format("Alpha, %f,\n", ALPHA);
            out.format("Gamma, %f,\n", GAMMA);
            out.format("Epsilon, %f,\n", EPSILON);
            switch(mCurrentLearningPolicy)
            {
                case NO_LEARNING_RANDOM:
                    out.format("Learning Policy, NO LEARNING RANDOM,\n");
                    break;
                case NO_LEARNING_GREEDY:
                    out.format("Learning Policy, NO LEARNING GREEDY,\n");
                    break;
                case SARSA:
                    out.format("Learning Policy, SARSA,\n");
                    break;
                case Q_LEARNING:
                    out.format("Learning Policy, Q LEARNING,\n");
                    break;
            }
            out.format("Intermediate Rewards, %b,\n", mIntermediateRewards);
            out.format("Terminal Rewards, %b,\n", mTerminalRewards);
            out.format("100 Rounds, Wins,\n");
            for (i = 0; i < getRoundNum()/100; i++)
            {
                out.format("%d, %d,\n", i + 1, mNumWinArray[i]);
            }

            out.format("Round, Average Delta Q,\n");
            for (i = 0; i < getRoundNum(); i++)
            {
                out.format("%d, %d,\n", i + 1, mAverageDeltaQ[i]);
            }

            out.close();
            fileOut.close();
        }
        catch (IOException exception)
        {
            exception.printStackTrace();
        }
    }
}