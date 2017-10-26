import java.io.File;
import java.io.IOException;
import java.lang.Math;
import java.util.Random;

/**
 * NeuralNet class for Assignment 1a of EECE592. A simple neural network with one hidden layer,
 * a selectable number of hidden neurons (up to MAX_HIDDEN_NEURONS) and a selectable number of
 * inputs (up to MAX_INPUTS)
 */
public class NeuralNet implements NeuralNetInterface
{
    // Constants
    static final int MAX_HIDDEN_NEURONS =  16;
    static final int MAX_INPUTS =          16;

    // Private member variables
    // Limits for custom sigmoid activation function used by the output neuron
    private double mArgA;
    private double mArgB;
    private double mWeightInitMin;
    private double mWeightInitMax;

    // Public member variables
    // Neural network parameters
    // We only have a single hidden layer with a provided number of inputs and hidden neurons
    public int mNumInputs;
    public int mNumHiddenNeurons;

    // Learning rate and momentum term
    public double mLearningRate;
    public double mMomentumTerm;

    // Array to store input values to the neural network, first index is bias input of 1.0
    public double[] mInputValues = new double[MAX_INPUTS];

    // Array to store input weights to the neurons of the hidden layer
    public static double[][] mInputWeights = new double[MAX_HIDDEN_NEURONS][MAX_INPUTS];
    // Array to store previous weights
    public static double[][] mPreviousInputWeights = new double[MAX_HIDDEN_NEURONS][MAX_INPUTS];

    // Array to store unactivated neuron outputs of the hidden layer
    public double[] mHiddenNeuronUnactivatedOutputs = new double[MAX_HIDDEN_NEURONS];
    // Array to store neuron outputs of the hidden layer
    public double[] mHiddenNeuronOutputs = new double[MAX_HIDDEN_NEURONS];
    // Array to store neuron errors of the hidden layer
    public double[] mHiddenNeuronErrors = new double[MAX_HIDDEN_NEURONS];

    // Array to store the output neuron's input weights
    public static double[] mOutputNeuronWeights = new double[MAX_HIDDEN_NEURONS];
    // Array to store the previous output neuron's weights
    public static double[] mPreviousOutputNeuronWeights = new double[MAX_HIDDEN_NEURONS];
    // Variables for output neuron bias weight
    public static double mOutputNeuronBiasWeight;
    public static double mPreviousOutputNeuronBiasWeight;

    // Variable for unactivated output neuron value
    public double mOutputNeuronUnactivatedValue;
    // Variable for value of output neuron
    public double mOutputNeuronValue;
    // Variable for out neuron error
    public double mOutputNeuronError;

    /**
     * Constructor for NeuralNet
     * @param argNumInputs The number of inputs in your input vector
     * @param argNumHidden The number of hidden neurons in your hidden layer. Only a single hidden layer is supported
     * @param argLearningRate The learning rate coefficient
     * @param argMomentumTerm The momentum coefficient
     * @param argA Integer lower bound of sigmoid used by the output neuron only.
     * @param argB Integer upper bound of sigmoid used by the output neuron only.
    */
    public NeuralNet(int argNumInputs,
                     int argNumHidden,
                     double argLearningRate,
                     double argMomentumTerm,
                     double argA,
                     double argB,
                     double argWeightInitMin,
                     double argWeightInitMax)
    {
        // Update our private variables
        mArgA = argA;
        mArgB = argB;
        mWeightInitMin = argWeightInitMin;
        mWeightInitMax = argWeightInitMax;
        // Add one here so that we don't worry about it later in the code (for bias)
        mNumInputs = argNumInputs + 1;
        mNumHiddenNeurons = argNumHidden;
        // Record the learning and momentum rates
        mLearningRate = argLearningRate;
        mMomentumTerm = argMomentumTerm;
        // Zero out the weights (also clears previous entry)
        zeroWeights();

//        System.out.format("Hi. Neural net instantiated with %5d inputs and %5d hidden neurons.\n", mNumInputs-1, mNumHiddenNeurons-1);
    }

    /**
     * This method implements the sigmoid function
     * @param x The input
     * @return f(x) = 1 / (1 + exp(-x))
     */
    public double sigmoid(double x)
    {
        double result;

        result =  1 / (1 + Math.exp(-x));

        return result;
    }

    /**
     * This method implements the first derivative of the sigmoid function
     * @param x The input
     * @return f'(x) = (1 / (1 + exp(-x)))(1 - (1 / (1 + exp(-x))))
     */
    public double sigmoidDerivative(double x)
    {
        double result;

        result = sigmoid(x)*(1 - sigmoid(x));

        return result;
    }

    /**
     * This method implements a general sigmoid with asymptotes bounded by (a,b)
     * @param x The input
     * @return f(x) = (b - a) / (1 + exp(-x)) + a
     */
    public double customSigmoid(double x)
    {
        double result;

        result = (mArgB - mArgA) * sigmoid(x) + mArgA;

        return result;
    }

    /**
     * This method implements the first derivative of the general sigmoid above
     * @param x The input
     * @return f'(x) = (1 / (b - a))(customSigmoid(x) - a)(b - customSigmoid(x))
     */
    public double customSigmoidDerivative(double x)
    {
        double result;

        result = (1.0/(mArgB - mArgA)) * (customSigmoid(x) - mArgA) * (mArgB - customSigmoid(x));

        return result;
    }

    /**
     * Initialize the weights to a random value between WEIGHT_INIT_MIN and WEIGHT_INIT_MAX
     */
    public void initializeWeights()
    {
        int i, j;

        // initialize inner neurons
        for(i = 0; i < mNumHiddenNeurons; i++)
        {
            for(j = 0; j < mNumInputs; j++)
            {
                mInputWeights[i][j] = getRandomDouble(mWeightInitMin, mWeightInitMax);
            }
            // initialize the output neuron weights
            mOutputNeuronWeights[i] = getRandomDouble(mWeightInitMin, mWeightInitMax);
            mOutputNeuronBiasWeight = getRandomDouble(mWeightInitMin, mWeightInitMax);
        }

        // Copy the initial weights into the delta tracking variables
        mPreviousInputWeights = mInputWeights.clone();
        mPreviousOutputNeuronWeights = mOutputNeuronWeights.clone();
        mPreviousOutputNeuronBiasWeight = mOutputNeuronBiasWeight;
    }

    public double calculateWeightDelta(double weightInput, double error, double currentWeight, double previousWeight)
    {
        double momentumTerm, learningTerm;

        momentumTerm = mMomentumTerm * (currentWeight - previousWeight);
        learningTerm = mLearningRate * error * weightInput;
        return (momentumTerm + learningTerm);
    }

    /**
     * Updates the weights based on the current backpropagated error
     */
    public void updateWeights()
    {
        int hiddenNeuron, input;
        double newOutputNeuronBiasWeight;
        double[] newOutputNeuronWeights = new double[MAX_HIDDEN_NEURONS];
        double[][] newInputNeuronWeights = new double[MAX_HIDDEN_NEURONS][MAX_INPUTS];

        // Update the weights to the output neuron
        // Update the bias input to the output neuron weight
        newOutputNeuronBiasWeight = mOutputNeuronBiasWeight + calculateWeightDelta(1.0, mOutputNeuronError, mOutputNeuronBiasWeight, mPreviousOutputNeuronBiasWeight);
        for(hiddenNeuron = 0; hiddenNeuron < mNumHiddenNeurons; hiddenNeuron++)
        {
            newOutputNeuronWeights[hiddenNeuron] = mOutputNeuronWeights[hiddenNeuron] +
                calculateWeightDelta(
                    mHiddenNeuronOutputs[hiddenNeuron],
                    mOutputNeuronError,
                    mOutputNeuronWeights[hiddenNeuron],
                    mPreviousOutputNeuronWeights[hiddenNeuron]);
        }


        // Update the weights to the hidden neurons
        for(hiddenNeuron = 0; hiddenNeuron < mNumHiddenNeurons; hiddenNeuron++)
        {
            for(input = 0; input < mNumInputs; input++)
            {
                newInputNeuronWeights[hiddenNeuron][input] = mInputWeights[hiddenNeuron][input] +
                    calculateWeightDelta(
                        mInputValues[input],
                        mHiddenNeuronErrors[hiddenNeuron],
                        mInputWeights[hiddenNeuron][input],
                        mPreviousInputWeights[hiddenNeuron][input]);
            }
        }

        mPreviousOutputNeuronBiasWeight = mOutputNeuronBiasWeight;
        mPreviousOutputNeuronWeights = mOutputNeuronWeights.clone();
        mPreviousInputWeights = mInputWeights.clone();

        mOutputNeuronBiasWeight = newOutputNeuronBiasWeight;
        mOutputNeuronWeights = newOutputNeuronWeights.clone();
        mInputWeights = newInputNeuronWeights.clone();
    }

    public void printNeuronOutputs()
    {
        int i;

        System.out.println("Current neuron outputs are as follows:");
        System.out.println("\tHidden neuron outputs");
        for(i = 0; i < mNumHiddenNeurons; i++)
        {
            System.out.format("%d, %5f\n", i, mHiddenNeuronOutputs[i]);
        }
        System.out.println("\tOutput Neuron Output");
        System.out.format("%5f\n", mOutputNeuronValue);
    }

    public void printNeuronErrors()
    {
        int i;

        System.out.println("Current neuron errors are as follows:");
        System.out.println("\tHidden neuron errors");
        for(i = 0; i < mNumHiddenNeurons; i++)
        {
            System.out.format("%d, %5f\n", i, mHiddenNeuronErrors[i]);
        }
        System.out.println("\tOutput neuron error");
        System.out.format("%5f\n", mOutputNeuronError);
    }

    public void printWeights()
    {
        int i, j;

        System.out.println("Current neuron weights are as follows:");
        System.out.println("\tHidden neuron weights");
        for(i = 0; i < mNumHiddenNeurons; i++)
        {
            for (j = 0; j < mNumInputs; j++)
            {
                System.out.format("%d %d %5f\n", i, j, mInputWeights[i][j]);
            }
        }
        System.out.println("\tOutput neuron weights");
        System.out.format("b %5f\n", mOutputNeuronBiasWeight);
        for(i = 0; i < mNumHiddenNeurons; i++)
        {
            System.out.format("%d %5f\n", i, mOutputNeuronWeights[i]);
        }
    }

    /**
     * Returns a random double value between specified min and max values
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

    public void zeroWeights()
    {
        int i, j;

        // initialize inner neurons
        for(i = 0; i < mNumHiddenNeurons; i++)
        {
            for(j = 0; j < mNumInputs; j++)
            {
                mInputWeights[i][j] = 0.0;
                mPreviousInputWeights[i][j] = 0.0;
            }
            // initialize the output neuron
            mPreviousOutputNeuronWeights[i] = 0.0;
            mOutputNeuronWeights[i] = 0.0;
        }
    }

    /**
     * @param x The input vector. An array of doubles.
     * @return The value returned by the NN for this input vector
     */
    public double outputFor(double[] x)
    {
        int hiddenNeuron, input;

        mInputValues = x;

        // Calculate hidden neuron outputs
        // Bias is included in input vector as the first index
        for(hiddenNeuron = 0; hiddenNeuron < mNumHiddenNeurons; hiddenNeuron++)
        {

            mHiddenNeuronUnactivatedOutputs[hiddenNeuron] = 0.0;
            // iterate over bias input + inputs
            for(input = 0; input < mNumInputs; input++)
            {
                mHiddenNeuronUnactivatedOutputs[hiddenNeuron] += mInputWeights[hiddenNeuron][input] * mInputValues[input];
            }
            // Apply the activation function to the weighted sum
            mHiddenNeuronOutputs[hiddenNeuron] = customSigmoid(mHiddenNeuronUnactivatedOutputs[hiddenNeuron]);
        }

        // Calculate the output of the output neuron
        mOutputNeuronUnactivatedValue = 0.0;
        for(hiddenNeuron = 0; hiddenNeuron < mNumHiddenNeurons; hiddenNeuron++)
        {
            mOutputNeuronUnactivatedValue += mHiddenNeuronOutputs[hiddenNeuron] * mOutputNeuronWeights[hiddenNeuron];
        }
        // Add the output bias
        mOutputNeuronUnactivatedValue += (1.0 * mOutputNeuronBiasWeight);
        // Apply the activation function to the weighted sum
        mOutputNeuronValue = customSigmoid(mOutputNeuronUnactivatedValue);

        return mOutputNeuronValue;
    }

    /**
     * This method calculates the error based on the current input & output.
     * It is expected that outputFor has been called before this method call.
     * @param expectedValue The expected output value for the current input
     */
    public void calculateErrors(double expectedValue)
    {
        int hiddenNeuron;

        // Calculate the output error from the feed forward
        mOutputNeuronError = (expectedValue - mOutputNeuronValue) * customSigmoidDerivative(mOutputNeuronUnactivatedValue);

        // Backpropagate the output error
        for(hiddenNeuron = 0; hiddenNeuron < mNumHiddenNeurons; hiddenNeuron++)
        {
            mHiddenNeuronErrors[hiddenNeuron] = mOutputNeuronError * mOutputNeuronWeights[hiddenNeuron] * customSigmoidDerivative(mHiddenNeuronUnactivatedOutputs[hiddenNeuron]);
        }
    }

    /**
     * This method will tell the NN the output
     * value that should be mapped to the given input vector. I.e.
     * the desired correct output value for an input.
     * @param x The input vector
     * @param argValue The new value to learn
     * @return The error in the output for that input vector
     */
    public double train(double[] x, double argValue)
    {
        // Feed forward stage: calculate the output value
        // this will update the neuron outputs
        outputFor(x);

        // Calculate errors
        calculateErrors(argValue);

        // perform weight update
        updateWeights();

        // Return the error in the output from what we expected
        return (argValue - mOutputNeuronValue);
    }

    public void save(File argFile)
    {

    }

    public void load(String argFileName) throws IOException
    {

    }
}