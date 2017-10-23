
public class TestBench
{
    public static void main(String[] args)
    {
        int MIN_VAL = 0;
        int MAX_VAL = 1;
        int NUM_INPUTS = 2;
        int NUM_HIDDEN_NEURONS = 4;
        double LEARNING_RATE = 0.5;
        double MOMENTUM = 0.5;

        int[][] XOR_TRAINING_SET = new int[][]{{0,0,0}, {0,1,1}, {1,0,1}, {1,1,0}};

        NeuralNet NeuralNetObj;

        NeuralNetObj = new NeuralNet(NUM_INPUTS, NUM_HIDDEN_NEURONS, LEARNING_RATE, MOMENTUM, MIN_VAL, MAX_VAL);
        NeuralNetObj.initializeWeights();
        NeuralNetObj = new NeuralNet(NUM_INPUTS, NUM_HIDDEN_NEURONS, LEARNING_RATE, MOMENTUM, MIN_VAL, MAX_VAL);
    }
}