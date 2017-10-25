import java.io.*;
import java.util.ArrayList;

public class TestBench
{
    // Training sets, binary and bipolar
    static final double[][] BIN_XOR_TRAINING_SET_IN = new double[][]
        {{ 1.0, 0.0, 0.0},
         { 1.0, 0.0, 1.0},
         { 1.0, 1.0, 0.0},
         { 1.0, 1.0, 1.0}};
    static final double[] BIN_XOR_TRAINING_SET_OUT = new double[]
        {  0.0,
           1.0,
           1.0,
           0.0};
    static final double[][] BIP_XOR_TRAINING_SET_IN = new double[][]
        {{ 1.0,-1.0,-1.0},
         { 1.0,-1.0, 1.0},
         { 1.0, 1.0,-1.0},
         { 1.0, 1.0, 1.0}};
    static final double[] BIP_XOR_TRAINING_SET_OUT = new double[]
        { -1.0,
           1.0,
           1.0,
          -1.0};

    // Constants for assignment questions
    // Trials to run to obtain convergence average
    static final int CONVERGENCE_AVERAGE_TRIALS = 500;
    // Number of epochs to test to before bailing
    static final int MAXIMUM_EPOCHS = 10000;
    // Number of NN inputs
    static final int NUM_INPUTS = 2;
    // Number of NN hidden neurons
    static final int NUM_HIDDEN_NEURONS = 4;
    // Squared error to b
    static final double CONVERGENCE_ERROR = 0.05;
    // 1a
    static final int MIN_VAL_1A = 0;
    static final int MAX_VAL_1A = 1;
    static final double MOMENTUM_1A = 0.0;
    static final double LEARNING_RATE_1A = 0.2;
    // 1b
    static final int MIN_VAL_1B = -1;
    static final int MAX_VAL_1B = 1;
    static final double MOMENTUM_1B = 0.0;
    static final double LEARNING_RATE_1B = 0.2;
    // 1c
    static final int MIN_VAL_1C = -1;
    static final int MAX_VAL_1C = 1;
    static final double MOMENTUM_1C = 0.9;
    static final double LEARNING_RATE_1C = 0.2;

    public static void main(String[] args) throws IOException
    {
        double epochAverage;
        NeuralNet neuralNetObj;
        ArrayList<Double> results = new ArrayList();

        try
        {
            // Print out a CSV to validate the sigmoid function implementations
            neuralNetObj = new NeuralNet(NUM_INPUTS, NUM_HIDDEN_NEURONS, LEARNING_RATE_1A, MOMENTUM_1A, MIN_VAL_1A, MAX_VAL_1A);
            printCustomSigmoidFunctions(neuralNetObj, "sigmoids.csv");

            // Part 1a
            // Define your XOR problem using a binary representation. Draw a graph of total error against number of
            // epochs. On average, how many epochs does it take to reach a total error of less than 0.05? You should
            // perform many trials to get your results, although you don’t need to plot them all.
            // Find out average number of epochs it takes to converge on binary XOR
            System.out.println("Starting 1a...");
            neuralNetObj = new NeuralNet(NUM_INPUTS, NUM_HIDDEN_NEURONS, LEARNING_RATE_1A, MOMENTUM_1A, MIN_VAL_1A, MAX_VAL_1A);
            epochAverage = runTrials(neuralNetObj, BIN_XOR_TRAINING_SET_IN, BIN_XOR_TRAINING_SET_OUT, CONVERGENCE_AVERAGE_TRIALS, CONVERGENCE_ERROR, MAXIMUM_EPOCHS, results);
            System.out.format("1a: %d successful trials to %1.2f total squared error convergence was average %1.3f\n", CONVERGENCE_AVERAGE_TRIALS, CONVERGENCE_ERROR, epochAverage);
            printTrialResults(results, "1a.csv");

            // Part 1b
            // This time use a bipolar representation. Again, graph your results to show the total error varying
            // against number of epochs. On average, how many epochs to reach a total error of less than 0.05?
            System.out.println("Starting 1b...");
            neuralNetObj = new NeuralNet(NUM_INPUTS, NUM_HIDDEN_NEURONS, LEARNING_RATE_1B, MOMENTUM_1B, MIN_VAL_1B, MAX_VAL_1B);
            epochAverage = runTrials(neuralNetObj, BIP_XOR_TRAINING_SET_IN, BIP_XOR_TRAINING_SET_OUT, CONVERGENCE_AVERAGE_TRIALS, CONVERGENCE_ERROR, MAXIMUM_EPOCHS, results);
            System.out.format("1b: %d successful trials to %1.2f total squared error convergence was average %1.3f\n", CONVERGENCE_AVERAGE_TRIALS, CONVERGENCE_ERROR, epochAverage);
            printTrialResults(results, "1b.csv");

            // Part 1c
            // Now set the momentum to 0.9. What does the graph look like now and how fast can 0.05 be reached?
            System.out.println("Starting 1c...");
            neuralNetObj = new NeuralNet(NUM_INPUTS, NUM_HIDDEN_NEURONS, LEARNING_RATE_1C, MOMENTUM_1C, MIN_VAL_1C, MAX_VAL_1C);
            epochAverage = runTrials(neuralNetObj, BIP_XOR_TRAINING_SET_IN, BIP_XOR_TRAINING_SET_OUT, CONVERGENCE_AVERAGE_TRIALS, CONVERGENCE_ERROR, MAXIMUM_EPOCHS, results);
            System.out.format("1c: %d successful trials to %1.2f total squared error convergence was average %1.3f\n", CONVERGENCE_AVERAGE_TRIALS, CONVERGENCE_ERROR, epochAverage);
            printTrialResults(results, "1c.csv");

        }
        catch (IOException e)
        {
            System.out.println(e);
        }
    }

    public static void printTrialResults(ArrayList<Double> results, String fileName) throws IOException
    {
        int epoch;
        PrintWriter printWriter = new PrintWriter(new FileWriter(fileName));
        printWriter.printf("Epoch, Total Squared Error,\n");
        for(epoch = 0; epoch < results.size(); epoch++)
        {
            printWriter.printf("%d, %f,\n", epoch, results.get(epoch));
        }
        printWriter.flush();
        printWriter.close();
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

    public static double runTrials(NeuralNet neuralNetObj, double[][] inVecs, double[] outVec, int numTrials, double convergenceError, int maxEpochs, ArrayList<Double> results)
    {
        int epochs, failedConvergences;
        int successfulTrials;
        double epochAverage;

        epochAverage = 0.0;
        successfulTrials = 0;
        failedConvergences = 0;
        do
        {
            // Clear our results
            results.clear();
            // Initialize weights for a new training session
            neuralNetObj.initializeWeights();
            // Attempt convergence
            epochs = attemptConvergence(
                neuralNetObj, inVecs, outVec, convergenceError, maxEpochs, results);
            // Check if we're under max epochs
            if(epochs < maxEpochs)
            {
                epochAverage += (double)epochs;
                successfulTrials++;
            }
            else
            {
                failedConvergences++;
                if(failedConvergences > 100000)
                {
                    break;
                }
            }
        }while(successfulTrials < numTrials);
        // Average out trials
        epochAverage /= successfulTrials;

        return epochAverage;
    }

    public static int attemptConvergence(NeuralNet NeuralNetObj, double[][] inVecs, double[] outVec, double convergenceError, int maxEpochs, ArrayList<Double> results)
    {
        double cummError;
        int index, epoch;

        for (epoch = 0; epoch < maxEpochs; epoch++)
        {
            cummError = 0.0;
            for (index = 0; index < 4; index++)
            {
                cummError += Math.pow(NeuralNetObj.train(inVecs[index], outVec[index]), 2.0);
            }

            // Append the result to our list
            results.add(cummError);

            if (cummError < convergenceError)
            {
                break;
            }
        }

        return epoch;
    }
}