package MyRobots;

import robocode.*;

import java.util.HashMap;

public class MettaBotLUT extends AdvancedRobot
{
    // Constants used in the robot
    public static final int ARENA_SIZEX_PX = 800;
    public static final int ARENA_SIZEY_PX = 600;

    // Variables to track the state of the arena
    public int mRobotX;
    public int mRobotY;
    public int mEnemyX;
    public int mEnemyY;

    // Hashmap to store state/action probabilities
    HashMap<Long, Double> mReinforcementLearningLUTHashMap = new HashMap<>();

    public void run()
    {

    }

    public void onScannedRobot(ScannedRobotEvent event)
    {
        // Obtain state information


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
     * This generates a hash for a given state. Everything is encoded in a long int (64 bits)
     * @return
     */
    public long generateStateHash(void)
    {
        long stateHash = 0;

        // Legend: [Max] -> Quantization -> Long int field mapping
        // Current position X [800] -> ->
        // Current position Y [600] -> ->
        // Distance between robot and opponent [1000] -> ->
        // Robot bearing [] -> ->
        // Gun bearing [] -> ->
        // Enemy bearing [] -> ->
        // Enemy gun bearing [] -> ->
        // Energy delta between robot and opponent [] -> ->

        return stateHash;
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
     * Updates a field in a long int
     * @param inputInteger The input long integer to modify
     * @param fieldWidth The width of the field to modify
     * @param fieldOffset The field's offset
     * @param value The value to update into the field
     * @return The updated input integer
     */
    public long updateLongIntField(long inputInteger, int fieldWidth, int fieldOffset, int value)
    {
        long mask;

        // Create mask
        mask = ~(((1<<fieldWidth) - 1) << fieldOffset);
        // Mask out field from input
        inputInteger &= mask;
        // OR in the new value into the field
        inputInteger |= value << fieldOffset;

        return inputInteger;
    }
}
