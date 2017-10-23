
public class TestBench
{
    public static void main(String[] args)
    {
        int MIN_VAL = 0;
        int MAX_VAL = 1;
        int NUM_INPUTS = 2;
        int NUM_HIDDEN_NEURONS = 4;
        int index, x1, x2;
        double outExp, outReal;
        double LEARNING_RATE = 0.5;
        double MOMENTUM = 0.5;

        int[][] XOR_TRAINING_SET = new int[][]{{0,0,0}, {0,1,1}, {1,0,1}, {1,1,0}};

        NeuralNet NeuralNetObj;

        NeuralNetObj = new NeuralNet(NUM_INPUTS, NUM_HIDDEN_NEURONS, LEARNING_RATE, MOMENTUM, MIN_VAL, MAX_VAL);
        NeuralNetObj.initializeWeights();
        NeuralNetObj = new NeuralNet(NUM_INPUTS, NUM_HIDDEN_NEURONS, LEARNING_RATE, MOMENTUM, MIN_VAL, MAX_VAL);

        for(index = 0; index < 4; index++)
        {
            x1 = XOR_TRAINING_SET[index][0];
            x2 = XOR_TRAINING_SET[index][1];
            outExp = (double)XOR_TRAINING_SET[index][2];
            outReal = NeuralNetObj.outputFor(new double[]{(double)x1, (double)x2});
            System.out.format("Output for %d %d: %5f (expect %5f)\n", x1, x2, outReal, outExp);
        }
    }
}