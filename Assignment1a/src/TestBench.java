import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

public class TestBench
{
    public static void main(String[] args) throws IOException
    {
        int MIN_VAL = -1;
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
             { 1.0, 1.0, 1.0}};
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

        // Print out a CSV to validate the sigmoid function implementations
//        NeuralNetObj = new NeuralNet(NUM_INPUTS, NUM_HIDDEN_NEURONS, LEARNING_RATE, MOMENTUM, MIN_VAL, MAX_VAL);
//        try
//        {
//            printCustomSigmoidFunctions(NeuralNetObj, "sigmoids.csv");
//        }
//        catch (IOException e)
//        {
//            System.out.println("Couldn't write to csv");
//        }

        // Part 1a
        // Instantiate the first object

        NeuralNetObj = new NeuralNet(NUM_INPUTS, NUM_HIDDEN_NEURONS, LEARNING_RATE, MOMENTUM, MIN_VAL, MAX_VAL);
        NeuralNetObj.initializeWeights();
//        NeuralNetObj.printWeights();
        for(epoch = 0; epoch < 100000; epoch++)
        {
            cummError = 0.0;
            for (index = 0; index < 4; index++)
            {
                inputVector = XOR_TRAINING_SET_IN[index];
                outExp = XOR_TRAINING_SET_OUT[index];
//                System.out.format("Input: %1f %1f %1f ", inputVector[0], inputVector[1], inputVector[2]);
                NeuralNetObj.outputFor(inputVector);
                outReal = NeuralNetObj.mOutputNeuronValue;
//                System.out.format("Output: %5f (expect %1f)\n",
//                outReal, outExp);

                instError = Math.pow(NeuralNetObj.train(inputVector, outExp), 2.0);
                cummError += instError;

            }
            cummError *= 0.5;
            if(cummError < 0.05)
            {
                System.out.format("Convered on epoch %d", epoch);
                break;
            }
        }
//        NeuralNetObj.printWeights();
    }

    public static void printCustomSigmoidFunctions(NeuralNet neuralNetObj, String fileName) throws IOException
    {
        double x, y, yPrime;

        PrintWriter printWriter = new PrintWriter(new FileWriter(fileName));

        printWriter.printf("x, Y(x), Y'(x),\n");
        for(x = -5.0; x <= 5.0; x += 0.1)
        {
            y = neuralNetObj.customSigmoid(x);
            yPrime = neuralNetObj.customSigmoidDerivative(x);
            printWriter.printf("%f, %f, %f\n", x, y, yPrime);
        }
        printWriter.close();
    }
}