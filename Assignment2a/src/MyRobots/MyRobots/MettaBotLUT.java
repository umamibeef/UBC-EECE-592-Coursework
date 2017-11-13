package MyRobots;

import robocode.*;

import java.awt.*;
import java.awt.event.KeyEvent;
import java.util.HashMap;
import java.util.Random;

import static java.awt.event.KeyEvent.*;

public class MettaBotLUT extends AdvancedRobot
{
    // Constants used in the robot
    public static final int ARENA_SIZEX_PX = 800;
    public static final int ARENA_SIZEY_PX = 600;
    public static final int ENEMY_ENERGY_THRESHOLD = 50;
    public static final boolean DEBUG_ON = true;
    public static final boolean DEBUG_OFF = false;
    // Move actions
    public static final int MOVE_UP = 0;
    public static final int MOVE_DN = 1;
    public static final int MOVE_LT = 2;
    public static final int MOVE_RT = 3;
    public static final int MOVE_DISTANCE = 50; // making this 10 allows for quantized moves
    // Fire actions
    public static final int FIRE_0  = 0;
    public static final int FIRE_1  = 1;
    public static final int FIRE_3  = 2;
    // Aim actions
    public static final int AIM_ST  = 0;
    public static final int AIM_LT  = 1;
    public static final int AIM_RT  = 2;
    public static final int AIM_MOD = 10; // add a degree offset to the aim
    // State hash field and offsets
    // TODO...
    // Action hash field and offsets
    public static final int ACTION_MOVE_OFFSET = 0;
    public static final int ACTION_MOVE_WIDTH = 2;
    public static final int ACTION_AIM_OFFSET = 2;
    public static final int ACTION_AIM_WIDTH = 2;
    public static final int ACTION_FIRE_OFFSET = 4;
    public static final int ACTION_FIRE_WIDTH = 2;

    // State variables
    public boolean mDebug = DEBUG_ON;

    // Variables to track the state of the arena
    public int mRobotX;
    public int mRobotY;
    public int mRobotHeading;
    public int mRobotGunHeading;
    public int mRobotEnergy;
    public int mEnemyDistance;
    public int mEnemyHeading;
    public int mEnemyBearing;
    public int mEnemyBearingFromGun;
    public int mEnemyEnergy;

    public Random mRandomInt = new Random();

    // Hashmap to store state/action probabilities
    HashMap<Integer, Double> mReinforcementLearningLUTHashMap = new HashMap<>();

    // Completion conditions
    TurnCompleteCondition mTurnComplete = new TurnCompleteCondition(this);
    MoveCompleteCondition mMoveComplete = new MoveCompleteCondition(this);
    GunTurnCompleteCondition mGunMoveComplete = new GunTurnCompleteCondition(this);

    public void run()
    {
        // Set colours
        setColors(Color.PINK, Color.PINK, Color.PINK, Color.PINK, Color.PINK);
        // Set robot properties
        setAdjustGunForRobotTurn(true);
        setAdjustRadarForGunTurn(true);
        setAdjustRadarForRobotTurn(true);

        waitFor(mTurnComplete);
        waitFor(mMoveComplete);
        waitFor(mGunMoveComplete);

        // Put robot in a known orientation
        turnGunRight(-getGunHeading());
        setTurnRight(-getHeading());
        execute();

        for(;;)
        {
            turnRadarRight(20);
        }
    }

    // Called when a key has been pressed
    public void onKeyPressed(java.awt.event.KeyEvent e)
    {
        int randomActionHash;

        switch (e.getKeyCode()) {
            case VK_R:
                // Perform a random action
                randomActionHash = mRandomInt.nextInt();
                parseActionHash(randomActionHash);
                execute();

                waitFor(mTurnComplete);
                waitFor(mMoveComplete);
                waitFor(mGunMoveComplete);
                break;
        }
    }

    // Called when a key has been released (after being pressed)
    public void onKeyReleased(KeyEvent e)
    {
        switch (e.getKeyCode())
        {
        }
    }

    public void onScannedRobot(ScannedRobotEvent event)
    {
        // Obtain state information
        // Robot's info
        mRobotX = (int)getX();
        mRobotY = (int)getY();
        mRobotHeading = (int)getHeading();
        mRobotGunHeading = (int)getGunHeading();
        mRobotEnergy = (int)getEnergy();
        // Enemy's info
        mEnemyDistance = (int)event.getDistance();
        mEnemyHeading = (int) event.getHeading();
        mEnemyBearing = (int)event.getBearing();
        mEnemyBearingFromGun = mRobotHeading - mRobotGunHeading + mEnemyBearing;
        mEnemyEnergy = (int)event.getEnergy();

        generateStateHash();

        if(mDebug)
        {
            System.out.format("Robot: X %d Y %d Heading %d GunHeading %d Energy %d\n",
            mRobotX, mRobotY, mRobotHeading, mRobotGunHeading, mRobotEnergy);
            System.out.format("Enemy: Distance %d Heading %d Bearing %d BearingFromGun %d Energy %d\n",
            mEnemyDistance, mEnemyHeading, mEnemyBearing, mEnemyBearingFromGun, mEnemyEnergy);
        }
    }

    public void onBattleEnded(BattleEndedEvent event)
    {

    }

    public void onDeath(DeathEvent event)
    {

    }

    public void onWin(WinEvent event)
    {

    }

    /**
     * This generates a hash for a given state. Everything is encoded in an int
     * @return a hash representing the current state
     */
    public int generateStateHash()
    {
        int stateHash = 0;

        int quantRobotX;
        int quantRobotY;
        int quantDistance;
        int quantRobotHeading;
        int quantEnemyBearingFromGun;
        int quantEnemyEnergy;

        // Legend: [max] -> quantization -> field width
        // Current position X                       [800]   -> 32   -> 5
        // Current position Y                       [600]   -> 32   -> 5
        // Distance between robot and opponent      [1000]  -> 32   -> 5
        // Robot bearing                            [360]   -> 32   -> 5
        // Enemy bearing                            [360]   -> 32   -> 5
        // Energy of enemy                          [N/A]   -> 2    -> 1

        // Quantization
        quantRobotX = quantizeInt(mRobotX, ARENA_SIZEX_PX, 32);
        quantRobotY = quantizeInt(mRobotY, ARENA_SIZEY_PX, 32);
        quantDistance = quantizeInt(mEnemyDistance, 1000, 32);
        quantRobotHeading = quantizeInt(mRobotHeading, 360, 32);
        quantEnemyBearingFromGun = quantizeInt(mEnemyBearingFromGun + 180, 360, 32);
        // For enemy energy, we will only care if it's above a threshold
        if(mEnemyEnergy > ENEMY_ENERGY_THRESHOLD)
        {
            quantEnemyEnergy = 0;
        }
        else
        {
            quantEnemyEnergy = 1;
        }

        // Assemble the hash
        stateHash = updateIntField(stateHash, 5, 0, quantRobotX);
        stateHash = updateIntField(stateHash, 5, 5, quantRobotY);
        stateHash = updateIntField(stateHash, 5, 10, quantDistance);
        stateHash = updateIntField(stateHash, 5, 15, quantRobotHeading);
        stateHash = updateIntField(stateHash, 5, 20, quantEnemyBearingFromGun);
        stateHash = updateIntField(stateHash, 1, 25, quantEnemyEnergy);

        if(mDebug)
        {
            System.out.format("Quantized values: %d %d %d %d %d %d\n",
                quantRobotX, quantRobotY, quantDistance, quantRobotHeading, quantEnemyBearingFromGun, quantEnemyEnergy);
            System.out.format("State hash: 0x%08x\n", stateHash);
        }

        return stateHash;
    }

    /**
     * This generates a hash for a given action. Everything is encoded in an int
     * @return
     */
    public int generateActionHash(int moveAction, int fireAction, int aimAction)
    {
        // Robot can do three things simultaneously:
        // Move up, down, left, or right                        (4)
        // Don't fire, fire 1, or fire 3                        (3)
        // Aim directly, or at some offset in either direction  (3)
        // 4 * 3 * 3 = 36 action possibilities, need at least 6 bits
        int actionHash = 0;

        actionHash = updateIntField(actionHash, ACTION_MOVE_WIDTH, ACTION_MOVE_OFFSET, moveAction);
        actionHash = updateIntField(actionHash, ACTION_FIRE_WIDTH, ACTION_FIRE_OFFSET, fireAction);
        actionHash = updateIntField(actionHash, ACTION_AIM_WIDTH, ACTION_AIM_OFFSET, aimAction);

        if(mDebug)
        {
            System.out.format("Action hash: 0x%08x\n", actionHash);
        }

        return actionHash;
    }

    /**
     * This parses the actions to take based on a given action hash
     * @param actionHash The action hash to parse into action
     */
    public void parseActionHash(int actionHash)
    {
        int moveDirection, fireType, aimType;

        moveDirection = getIntFieldVal(actionHash, ACTION_MOVE_WIDTH, ACTION_MOVE_OFFSET);
        fireType = getIntFieldVal(actionHash, ACTION_FIRE_WIDTH, ACTION_FIRE_OFFSET);
        aimType = getIntFieldVal(actionHash, ACTION_AIM_WIDTH, ACTION_AIM_OFFSET);

        // Perform the move action
        switch(moveDirection)
        {
            case MOVE_UP:
                setTurnRight(-getHeading());
                execute();
                waitFor(mTurnComplete);
                setAhead(MOVE_DISTANCE);
                break;
            case MOVE_DN:
                setTurnRight(-getHeading()+180);
                execute();
                waitFor(mTurnComplete);
                setAhead(MOVE_DISTANCE);
                break;
            case MOVE_LT:
                setTurnRight(-getHeading()+270);
                execute();
                waitFor(mTurnComplete);
                setAhead(MOVE_DISTANCE);
                break;
            case MOVE_RT:
                setTurnRight(-getHeading()+90);
                execute();
                waitFor(mTurnComplete);
                setAhead(MOVE_DISTANCE);
                break;
            default:
                // We should never be in here, do nothing.
                break;
        }

        // Perform the firing type action
        switch(fireType)
        {
            case FIRE_0:
                // We don't fire in this case
                break;
            case FIRE_1:
                // Fire a 1 power bullet
                setFireBullet(1.0);
                break;
            case FIRE_3:
                // Fire a 3 power bullet
                setFireBullet(3.0);
                break;
            default:
                // We should never be in here, do nothing.
                break;
        }

        // Perform the aim type action
        if(mEnemyBearing != 0)
        {
            switch(aimType)
            {
                case AIM_ST:
                    // Aim directly for the enemy
                    setTurnGunRight(mEnemyBearingFromGun);
                    break;
                case AIM_LT:
                    // Aim to the left of the enemy by a modifier
                    setTurnGunRight(mEnemyBearingFromGun - AIM_MOD);
                    break;
                case AIM_RT:
                    // Aim to the right of the enemy by a modifier
                    setTurnGunRight(mEnemyBearingFromGun + AIM_MOD);
                    break;
                default:
                    // We should never be in here, do nothing.
                    break;
            }
        }
    }

    /**
     * Returns the distance between two coordinates
     * @param x0 X position of the first coordinate
     * @param y0 Y position of the first coordinate
     * @param x1 X position of the second coordinate
     * @param y1 Y position of the second coordinate
     * @return A double value representing the distance between two coordinates
     */
    public double calculateDistance(int x0, int y0, int x1, int y1)
    {
        double distance;

        distance = Math.sqrt(Math.pow((x1 - x0), 2) + Math.pow((y1 - y0), 2));

        return distance;
    }

    /**
     * Updates a field in an int
     * @param inputInteger The input integer to modify
     * @param fieldWidth The width of the field to modify
     * @param fieldOffset The field's offset
     * @param value The value to update into the field
     * @return The updated input integer
     */
    public int updateIntField(int inputInteger, int fieldWidth, int fieldOffset, int value)
    {
        int returnValue;
        int mask;

        returnValue = inputInteger;

        // Create mask
        mask = ~(((1<<fieldWidth) - 1) << fieldOffset);
        // Mask out field from input
        returnValue &= mask;
        // OR in the new value into the field
        returnValue |= value << fieldOffset;

        return returnValue;
    }

    /**
     * Returns the value of a field in an int
     * @param inputInteger The input integer to extract the value from
     * @param fieldWidth The width of the field to extract
     * @param fieldOffset The offset of the field to extract
     * @return
     */
    public int getIntFieldVal(int inputInteger, int fieldWidth, int fieldOffset)
    {
        int returnValue;
        int mask;

        returnValue = inputInteger;

        // Create mask
        mask = ((1<<fieldWidth) - 1) << fieldOffset;
        // Mask out the field from the input
        returnValue &= mask;
        // Shift down to grab it
        returnValue >>= fieldOffset;

        return returnValue;
    }

    /**
     * Quantizes an integer based on its current max and the desired quantization max
     * @param value The value to be quantized
     * @param realMax The actual maximum of the value
     * @param quantizedMax The desired quantized maximum for the value
     * @return The quantized version of the value
     */
    public int quantizeInt(int value, int realMax, int quantizedMax)
    {
        int quantizedVal;

        quantizedVal = (int)((double)value * (double)quantizedMax / (double)realMax);

        return quantizedVal;
    }
}
