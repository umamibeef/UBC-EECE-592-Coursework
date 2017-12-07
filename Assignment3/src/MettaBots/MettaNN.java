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
    private static final int NUM_INPUTS = 4;            // Number of NN inputs
    private static final int NUM_HIDDEN_NEURONS = 200;  // Number of NN hidden neurons
    private static final int NUM_OUTPUTS = 8;           // Number of NN outputs
    private static final int MIN_VAL = -1;              // Minimum value for activation function (sigmoid)
    private static final int MAX_VAL = 1;               // Maximum value for activation function (sigmoid)
    private static final double MOMENTUM = 0.2;         // Momentum parameter for backpropagation
    private static final double LEARNING_RATE = 0.005;  // Learning rate parameter for backpropagation
    private static final double WEIGHT_INIT_MIN = -2.0; // Random weight init low limit
    private static final double WEIGHT_INIT_MAX = 1.0;  // Random weight init high limit

    // Reinforcement learning parameters
    private static final double ALPHA = 0.5;    // Fraction of difference used
    private static final double GAMMA = 0.8;    // Discount factor
    private static final double EPSILON = 0.1;  // Probability of exploration
    //private int mCurrentLearningPolicy = NO_LEARNING_RANDOM;
    //private int mCurrentLearningPolicy = NO_LEARNING_GREEDY;
    private int mCurrentLearningPolicy = SARSA;
    //private int mCurrentLearningPolicy = Q_LEARNING;
    private boolean mIntermediateRewards = true;
    private boolean mTerminalRewards = true;

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
    // Action constants
    private static final int ACTION_DIMENSIONALITY = ACTION_MOVE_NUM * ACTION_FIRE_NUM;
    private static final int ACTION_MODE_MAX_Q = 0;
    private static final int ACTION_MODE_EPSILON_GREEDY = 1;

    // State constants
    private static final int STATE_POS_X_MAX = ARENA_SIZEX_PX;
    private static final int STATE_POS_Y_MAX = ARENA_SIZEY_PX;
    private static final int STATE_DISTANCE_MAX = 1000;
    private static final int STATE_ROBOT_HEADING_MAX = 360;
    private static final int STATE_DIMENSIONALITY = 4; // Total number of state dimensions
    private static final int STATE_POS_X_INDEX = 0;
    private static final int STATE_POS_Y_INDEX = 1;
    private static final int STATE_DISTANCE_INDEX = 2;
    private static final int STATE_HEADING_INDEX = 3;

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
    private double [] mCurrrentStateSnapshot = new double[STATE_DIMENSIONALITY];
    private double [] mPreviousStateSnapshot = new double[STATE_DIMENSIONALITY];

    // Variables for learning
    private int mPreviousStateActionHash;
    private int mCurrentStateActionHash = NULL_32;

    private int mPreviousEnergyDifference;
    private int mCurrentEnergyDifference;
    private double mCurrentReward;

    // Neural network to approximate Q(s,a) function
    private static NeuralNetMulti mNeuralNet;
    private static ArrayList<ArrayList<ArrayList<Double>>> mNeuralNetWeights;

    // Completion conditions
    private final TurnCompleteCondition mTurnComplete = new TurnCompleteCondition(this);
    private final MoveCompleteCondition mMoveComplete = new MoveCompleteCondition(this);
    private final GunTurnCompleteCondition mGunMoveComplete = new GunTurnCompleteCondition(this);

    // Winrate tracking for every 100 rounds
    private static final int NUM_ROUNDS = 100000;
    private static final int NUM_ROUNDS_DIV_100 = NUM_ROUNDS / 100;
    private static int [] mNumWinArray = new int[NUM_ROUNDS_DIV_100];

    public void run()
    {
        int currentStateHash, selectedAction;
        long fileSize;
        ArrayList<ArrayList<ArrayList<Double>>> neuralNetworkWeights;

        // Instantiate a new neural network for the robot to learn
        mNeuralNet = new NeuralNetMulti(
            NUM_INPUTS, NUM_OUTPUTS, NUM_HIDDEN_NEURONS, LEARNING_RATE, MOMENTUM, MIN_VAL, MAX_VAL, WEIGHT_INIT_MIN, WEIGHT_INIT_MAX);
        // Initialize weights for a new training session
        mNeuralNet.initializeWeights();

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
        printDebug("Current neural network file size ");
        // If the current file is empty, put in an empty hashmap
        if (fileSize == 0)
        {
            newWeightFile(mNeuralNetWeightsFile);
        }

        printDebug("Data available: %d bytes\n", getDataQuotaAvailable());

        // If SARSA, we must take an action at start
        // Choose an action hash that has the maximum Q for this state
        if(mCurrentLearningPolicy == SARSA)
        {
            // Take a snapshot of the current state
            mCurrrentStateSnapshot = takeStateSnapshot();
            // Take an action based on the current state
            takeAction(ACTION_MODE_EPSILON_GREEDY, mCurrrentStateSnapshot);
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

        printDebug("Robot: X %d Y %d Heading %d GunHeading %d Energy %d\n",
        mRobotX, mRobotY, mRobotHeading, mRobotGunHeading, mRobotEnergy);
        printDebug("Enemy: Distance %d Heading %d Bearing %d BearingFromGun %d Energy %d\n",
        mEnemyDistance, mEnemyHeading, mEnemyBearing, mEnemyBearingFromGun, mEnemyEnergy);

        learn(NON_TERMINAL_STATE);
    }

    private void learn(boolean terminalState)
    {
        double qPrevNew;
        int actionHash;
        double [] stateArray;

        printDebug("==[LEARN]=========================================\n");

        // Determine current state input at the time of learn()
        stateArray = generateStateArray();

        // Calculate the current reward
        // Reward can be obtained asynchronously through events such as bullet hitting
        // Add to the current reward the CHANGE in difference between previous and current
        if (mIntermediateRewards)
        {
            mPreviousEnergyDifference = mCurrentEnergyDifference;
            mCurrentEnergyDifference = mRobotEnergy - mEnemyEnergy;
            mCurrentReward += mCurrentEnergyDifference - mPreviousEnergyDifference;
        }
        printDebug("Current reward: %f\n", mCurrentReward);

        switch (mCurrentLearningPolicy)
        {
            // No learning at all (baseline)
            case NO_LEARNING_RANDOM:
                printDebug("No learning and random!\n");
                // Take random action
                actionHash = getRandomAction();
                printDebug("Taking random action of 0x%02x\n", actionHash);
                takeAction(stateArray, actionHash);
                break;
            case NO_LEARNING_GREEDY:
                printDebug("No learning and totally greedy!\n");
                // Observe the new environment
                stateArray = generateStateArray();
                // Get the action hash that has the maximum Q for this state
                printDebug("Find the maximum Q action for this new state");
                actionHash = getActionHash(ACTION_MODE_MAX_Q, stateArray);
                // Take the action
                takeAction(stateArray, actionHash);
                break;
            // On-policy (SARSA)
            case SARSA:
                printDebug("SARSA!\n");
                // Choose an on-policy action
                actionHash = getActionHash(ACTION_MODE_EPSILON_GREEDY, stateArray);
                // Calculate new value for previous Q;
                qPrevNew = calculateQPrevNew(getQValue(combineStateActionHashes(stateArray, actionHash)));

                // Update the LUT with the new value for the previous Q
                mReinforcementLearningLUTHashMap.put(mPreviousStateActionHash, qPrevNew);

                // Reset reward until the next learn
                mCurrentReward = 0.0;
                // We're done! No more actions.
                if (terminalState)
                {
                    printDebug("Terminal state! No more actions available.\n");
                    return;
                }
                // Take the next action
                takeAction(stateArray, actionHash);
                break;
            // Off-policy (Q-Learning)
            case Q_LEARNING:
                printDebug("Q-learning!\n");
                // We're done! No more actions.
                if (terminalState)
                {
                    printDebug("Terminal state! No more actions available.\n");
                    // Learn from the terminal action
                    // Calculate new value for previous Q, next Q is zero since we are terminal
                    qPrevNew = calculateQPrevNew(0.0);

                    // Update the LUT with the new value for the previous Q
                    mReinforcementLearningLUTHashMap.put(mPreviousStateActionHash, qPrevNew);

                    return;
                }
                else
                {
                    // Choose an on-policy action
                    printDebug("Choosing on-policy action to take");
                    actionHash = getActionHash(ACTION_MODE_EPSILON_GREEDY, stateArray);
                    // Take the action
                    takeAction(stateArray, actionHash);
                    // Observe the new environment
                    stateArray = generateStateArray();
                    // Get the action hash that has the maximum Q for this state
                    printDebug("Find the maximum Q action for this new state");
                    actionHash = getActionHash(ACTION_MODE_MAX_Q, stateArray);
                    // Calculate new value for previous Q;
                    qPrevNew = calculateQPrevNew(getQValue(combineStateActionHashes(stateArray, actionHash)));

                    // Update the LUT with the new value for the previous Q
                    mReinforcementLearningLUTHashMap.put(mPreviousStateActionHash, qPrevNew);

                    // Reset reward until the next learn
                    mCurrentReward = 0.0;
                }
                break;
            default:
                break;
        }
    }

    /**
     * This function will either explore randomly or take an 1-EPSILON greedy action that is passed into it
     * @param currentStateHash The current state hash
     * @param actionHash The action to take
     */
    private void takeAction(int currentStateHash, double [] stateSnapshot)
    {
        // Update current state/action hash
        mCurrentStateActionHash = updateIntField(currentStateHash, ACTION_FIELD_WIDTH, ACTION_FIELD_OFFSET, actionHash);
        // Parse action hash
        parseActionHash(actionHash);
        // Update the previous action hash
        mPreviousStateActionHash = mCurrentStateActionHash;
    }

    /**
     * Updates the previous Q value based on the passed in next Q value and the learning hyperparameters
     * @param qNext The next Q value, could be max or could be chosen on-policy
     * @return
     */
    private double calculateQPrevNew(double qNext)
    {
        double qPrevNew, qPrevOld;

        qPrevOld = getQValue(mPreviousStateActionHash);
        qPrevNew = qPrevOld + (ALPHA * (mCurrentReward + (GAMMA * qNext) - qPrevOld));

        return qPrevNew;
    }

    /**
     * Obtain the action hash with the largest Q value based on the provided action hash.
     * If multiple actions are tied for the largest Q value, pick one at random.
     *
     * @param mode The action selection mode, either epsilon greedy or qmax
     * @param currentStateHash Get the action with the maximum Q from the provided state hash
     * @return The action with the highest Q for the current state hash
     */
    private int getActionHash(int mode, int currentStateHash)
    {
        int moveAction, fireAction;
        int randomPick, actionHash, completeHash, selectedActionHash, selectedCompleteHash;
        int[] qMaxActions;
        int currentQMaxActionNum = 0;
        double currentMax = -999.0;
        double qVal, randomDouble;

        qMaxActions = new int[ACTION_DIMENSIONALITY];
        // Iterate through all possible actions
        printDebug("Current state hash: 0x%08x\n", currentStateHash);
        printDebug("Possible Q-values:\n");
        for (moveAction = 0; moveAction < ACTION_MOVE_NUM; moveAction++)
        {
            for (fireAction = 0; fireAction < ACTION_FIRE_NUM; fireAction++)
            {
                // Calculate the action hash and create the complete hash by adding it to the current state hash
                actionHash = generateActionHash(moveAction, fireAction);
                completeHash = combineStateActionHashes(currentStateHash, actionHash);
                printDebug("0x%08x: %f\n", completeHash, getQValue(completeHash));

                qVal = getQValue(completeHash);

                // Update current max
                if (qVal > currentMax)
                {
                    // New max, clear array
                    // We can have a maximum of the number of possible actions as the number of possible actions
                    qMaxActions = new int[ACTION_DIMENSIONALITY];
                    currentQMaxActionNum = 1;
                    qMaxActions[currentQMaxActionNum - 1] = completeHash;
                    currentMax = qVal;
                }
                else if (qVal == currentMax)
                {
                    currentQMaxActionNum++;
                    qMaxActions[currentQMaxActionNum - 1] = completeHash;
                }
            }
        }

        printDebug("Max actions: %d\n", currentQMaxActionNum);
        if (currentQMaxActionNum == 1)
        {
            selectedCompleteHash = qMaxActions[0];
            selectedActionHash = getIntFieldVal(selectedCompleteHash, ACTION_FIELD_WIDTH, ACTION_FIELD_OFFSET);
            printDebug("Found best possible action to take [0x%02x] with Q-value of %f\n",
            selectedActionHash, getQValue(selectedCompleteHash));
        }
        else
        {
            randomPick = getRandomInt(0, currentQMaxActionNum - 1);
            selectedCompleteHash = qMaxActions[randomPick];
            selectedActionHash = getIntFieldVal(selectedCompleteHash, ACTION_FIELD_WIDTH, ACTION_FIELD_OFFSET);
            printDebug("Found %d possible actions to take, randomly picking index %d [0x%02x] with Q-value of %f\n",
            currentQMaxActionNum, randomPick, selectedActionHash, getQValue(selectedCompleteHash));
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
                    printDebug("Got random number %f\n", randomDouble);
                    // Take random action
                    selectedActionHash = getRandomAction();
                    printDebug("Picking random action of 0x%02x\n", selectedActionHash);
                }
                else
                {
                    // Take greedy action
                    printDebug("Picking greedy action of 0x%02x\n", selectedActionHash);
                }
                break;
            // We should already have max Q from above, so choose that
            case ACTION_MODE_MAX_Q:
                printDebug("Picking action of 0x%02x\n", selectedActionHash);
                break;
            default:
                // We should never be here
                break;
        }

        return selectedActionHash;
    }

    public void onBulletHit(BulletHitEvent event)
    {
        if (mIntermediateRewards)
        {
            mCurrentReward += 30;
        }
    }

    public void onHitByBullet(HitByBulletEvent event)
    {
        if (mIntermediateRewards)
        {
            mCurrentReward -= 30;
        }
    }

    public void onBattleEnded(BattleEndedEvent event)
    {
        // Save the LUT to the data file
        saveLut(mLutFile);
        // Save the win tracker to the tracker file
        saveStats(mStatsFile);
    }

    public void onDeath(DeathEvent event)
    {

        // Give terminal reward of -100
        if (mTerminalRewards)
        {
            mCurrentReward -= 100;
            learn(TERMINAL_STATE);
        }
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

    private double[] getQValue(ArrayList<Double> preprocessedState)
    {
        int i;
        double[] state = new double[preprocessedState.size()];
        double[] qValues;

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

        // Quantization
        quantRobotX = quantizeInt(mRobotX, STATE_POS_X_MAX, 1<<STATE_POS_X_WIDTH);
        quantRobotY = quantizeInt(mRobotY, STATE_POS_Y_MAX, 1<<STATE_POS_Y_WIDTH);
        quantDistance = quantizeInt(mEnemyDistance, STATE_DISTANCE_MAX, 1<<STATE_DISTANCE_WIDTH);
        quantRobotHeading = quantizeInt(mRobotHeading, STATE_ROBOT_HEADING_MAX, 1<<STATE_ROBOT_HEADING_WIDTH);

        // Assemble the hash
        stateHash = updateIntField(stateHash, STATE_POS_X_WIDTH, STATE_POS_X_OFFSET, quantRobotX);
        stateHash = updateIntField(stateHash, STATE_POS_Y_WIDTH, STATE_POS_Y_OFFSET, quantRobotY);
        stateHash = updateIntField(stateHash, STATE_DISTANCE_WIDTH, STATE_DISTANCE_OFFSET, quantDistance);
        stateHash = updateIntField(stateHash, STATE_ROBOT_HEADING_WIDTH, STATE_ROBOT_HEADING_OFFSET, quantRobotHeading);

        //printDebug("Quantized values: %d %d %d %d %d %d\n",
        //    quantRobotX, quantRobotY, quantDistance, quantRobotHeading, quantEnemyBearingFromGun, quantEnemyEnergy);
        printDebug("Quantized values: %d %d %d %d\n",
        quantRobotX, quantRobotY, quantDistance, quantRobotHeading);
        printDebug("State hash: 0x%08x\n", stateHash);

        // Check if any values are negative, something went wrong
        if ((quantRobotX < 0) || (quantRobotY < 0) || (quantDistance < 0) || (quantRobotHeading < 0))
        {
            throw new ArithmeticException("Quantized value cannot be negative!!!");
        }

        return stateHash;
    }

    /**
     * This generates a hash for a given action. Everything is encoded in an int
     *
     * @return Returns a hash based on the selected action
     */
    private int generateActionHash(int moveAction, int fireAction)
    {
        // Robot can do two things simultaneously:
        // Move up, down, left, or right                        (4)
        // Don't fire or fire 3                                 (2)
        // 4 * 2 = 8 action possibilities, need at least 3 bits
        int actionHash = 0;

        actionHash = updateIntField(actionHash, ACTION_MOVE_WIDTH, ACTION_MOVE_OFFSET, moveAction);
        actionHash = updateIntField(actionHash, ACTION_FIRE_WIDTH, ACTION_FIRE_OFFSET, fireAction);

        //printDebug("Action hash: 0x%08x\n", actionHash);

        return actionHash;
    }

    /**
     * This parses the actions to take based on a given action hash.
     *
     * @param actionHash The action hash to parse into action
     */
    private void parseActionHash(int actionHash)
    {
        int moveDirection, fireType, newHeading, estimatedEnemyBearingFromGun;
        double angle;

        moveDirection = getIntFieldVal(actionHash, ACTION_MOVE_WIDTH, ACTION_MOVE_OFFSET);
        fireType = getIntFieldVal(actionHash, ACTION_FIRE_WIDTH, ACTION_FIRE_OFFSET);

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
        angle = absoluteBearing((int) getX(), (int) getY(), mEnemyX, mEnemyY);
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
    }

    /**
     * Get a random action by randomizing an action hash
     */
    private int getRandomAction()
    {
        int actionHash, randomMove, randomFire;//, randomAim;

        randomMove = getRandomInt(0, ACTION_MOVE_NUM - 1);
        randomFire = getRandomInt(0, ACTION_FIRE_NUM - 1);

        actionHash = generateActionHash(randomMove, randomFire);

        return actionHash;
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
    private double absoluteBearing(int x0, int y0, int x1, int y1)
    {
        int xo = x1 - x0;
        int yo = y1 - y0;
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
    private double calculateDistance(int x0, int y0, int x1, int y1)
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
    private double scaleValue(double value, double valMax, double targetMin, double targetMax)
    {
        double scaledVal, range;

        range = targetMax - targetMin;

        scaledVal = value * range / valMax;

        return scaledVal;
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

            out.close();
            fileOut.close();
        }
        catch (IOException exception)
        {
            exception.printStackTrace();
        }
    }
}