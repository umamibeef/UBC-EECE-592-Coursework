
public class TestBench
{
    public static void main(String[] args)
    {
        int MIN_VAL = 0;
        int MAX_VAL = 1;
        int NUM_INPUTS = 2;
        int NUM_HIDDEN_NEURONS = 4;
        int index, epoch;
        double x1, x2, outExp, outReal, instError, cummError;
        double LEARNING_RATE = 0.2;
        double MOMENTUM = 0.0;

        double[][] XOR_TRAINING_SET = new double[][]{{0.0,0.0,0.0}, {0.0,1.0,1.0}, {1.0,0.0,1.0}, {1.0,1.0,0.0}};

        NeuralNet NeuralNetObj;

        // Part 1a
        // Instantiate the first object
        NeuralNetObj = new NeuralNet(NUM_INPUTS, NUM_HIDDEN_NEURONS, LEARNING_RATE, MOMENTUM, MIN_VAL, MAX_VAL);
        NeuralNetObj.initializeWeights();
        NeuralNetObj.printNeuronWeights();

        epoch = 0;

        do
        {
            cummError = 0.0;

            System.out.format("*** EPOCH %d ***\n", epoch++);
            for(index = 0; index < 4; index++)
            {
                x1 = XOR_TRAINING_SET[index][0];
                x2 = XOR_TRAINING_SET[index][1];
                outExp = XOR_TRAINING_SET[index][2];
                instError = Math.pow(NeuralNetObj.train(new double[]{x1, x2}, XOR_TRAINING_SET[index][2]), 2.0);
                cummError += instError;
                outReal = NeuralNetObj.mOutputNeuronValue;
                System.out.format("Output for %1f %1f: %5f (expect %5f, error %5f)\n",
                x1, x2, outReal, outExp, instError);
            }

            cummError *= 0.5;
        }
        while(epoch < 100000);
        //while(Math.abs(cummError)>0.05);
    }
}