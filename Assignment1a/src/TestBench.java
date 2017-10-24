
public class TestBench
{
    public static void main(String[] args)
    {
        int MIN_VAL = 0;
        int MAX_VAL = 1;
        int NUM_INPUTS = 2;
        int NUM_HIDDEN_NEURONS = 4;
        int index, epoch, convergence;
        double inputVector[], outExp, outReal, instError, cummError;
        double LEARNING_RATE = 0.2;
        double MOMENTUM = 0.0;

        NeuralNet NeuralNetObj;

        double[][] XOR_TRAINING_SET_IN;
        double[] XOR_TRAINING_SET_OUT;

        double[][] BIN_XOR_TRAINING_SET_IN = new double[][]
            {{ 1.0, 0.0, 0.0},
             { 1.0, 0.0, 1.0},
             { 1.0, 1.0, 0.0},
             { 1.0, 1.0, 0.0}};
        double[] BIN_XOR_TRAINING_SET_OUT = new double[]
            { 0.0,
              1.0,
              1.0,
              0.0};
        double[][] BIP_XOR_TRAINING_SET_IN = new double[][]
            {{ 1.0,-1.0,-1.0},
             { 1.0,-1.0, 1.0},
             { 1.0, 1.0,-1.0},
             { 1.0, 1.0, 1.0}};
        double[] BIP_XOR_TRAINING_SET_OUT = new double[]
            {-1.0,
              1.0,
              1.0,
             -1.0};

        if(MIN_VAL == 0.0)
        {
            XOR_TRAINING_SET_IN = BIN_XOR_TRAINING_SET_IN;
            XOR_TRAINING_SET_OUT = BIN_XOR_TRAINING_SET_OUT;
        }
        else
        {
            XOR_TRAINING_SET_IN = BIP_XOR_TRAINING_SET_IN;
            XOR_TRAINING_SET_OUT = BIP_XOR_TRAINING_SET_OUT;
        }

        // Part 1a
        // Instantiate the first object

        convergence = 0;

        for(;;)
        {
            NeuralNetObj = new NeuralNet(NUM_INPUTS, NUM_HIDDEN_NEURONS, LEARNING_RATE, MOMENTUM, MIN_VAL, MAX_VAL);
            NeuralNetObj.initializeWeights();

            epoch = 0;

            do
            {

                cummError = 0.0;

                for (index = 0; index < 4; index++)
                {
                    inputVector = XOR_TRAINING_SET_IN[index];
                    outExp = XOR_TRAINING_SET_OUT[index];

                    NeuralNetObj.outputFor(inputVector);
//                    outReal = NeuralNetObj.mOutputNeuronValue;
//                                    System.out.format("Output for %1f %1f: %5f (expect %5f)\n",
//                                    inputVector[1], inputVector[2], outReal, outExp);

                    instError = Math.pow(NeuralNetObj.train(inputVector, outExp), 2.0);
                    cummError += instError;

                }
                cummError *= 0.5;
                System.out.println(cummError);
                epoch++;
            }
            while(Math.abs(cummError) > 0.05 && epoch < 200000);
//            System.out.format("Bailed at %d epochs\n", epoch);
            if(epoch < 200000 && Math.abs(cummError) < 0.05)
            {
                System.out.format("*** CONVERGED ON EPOCH %d ***\n", epoch);
                convergence++;
            }

            if(convergence == 20)
            {
                break;
            }
        }
    }
}