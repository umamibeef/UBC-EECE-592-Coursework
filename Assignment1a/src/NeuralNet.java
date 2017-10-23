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
    static final int MAX_HIDDEN_NEURONS =  256;
    static final int MAX_INPUTS =          10;
    static final int FIRST_INPUT_INDEX =   1;
    static final int BIAS_INPUT_INDEX =    0;
    static final int FIRST_WEIGHT_INDEX =  1;
    static final int BIAS_WEIGHT_INDEX =   0;
    static final double WEIGHT_INIT_MIN =  -0.5;
    static final double WEIGHT_INIT_MAX =  0.5;

    // Private member variables
    // Limits for custom sigmoid activation function used by the output neuron
    private double mArgA;
    private double mArgB;

    // Public member variables
    // Neural network parameters
    // We only have a single hidden layer with a provided number of inputs and hidden neurons
    public int mNumInputs;
    public int mNumHiddenNeurons;
    // Array to store input values to the neural network, first index is bias input of 1
    public double[] mInputValues = new double[MAX_INPUTS];
    // Array to store neuron weights of hidden layer
    public static double[] mNeuronWeights = new double[MAX_HIDDEN_NEURONS];
    // Array to store neuron outputs of the hidden layer
    public double[] mNeuronOutputs = new double[MAX_HIDDEN_NEURONS];
    // Variable for the value of the output neuron's weight
    public static double mOutputNeuronWeight;
    // Variable for value of output neuron
    public double mOutputValue;

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
                              double argB)
    {
        // Update our private variables
        mArgA = argA;
        mArgB = argB;
        // Add one here so that we don't worry about it later in the code
        mNumInputs = argNumInputs+1;
        mNumHiddenNeurons = argNumHidden+1;

        // Update the bias value to one
        mInputValues[BIAS_INPUT_INDEX] = 1.0;

        System.out.format("Hi. Neural net instantiated with %d inputs and %d hidden neurons.\n", mNumInputs-1, mNumHiddenNeurons-1);

        // Print out neuron weights
        printNeuronWeights();
    }

    /**
     * This method implements the sigmoid function
     * @param x The input
     * @return f(x) = 1 / (1 + exp(-x))
     */
    public double sigmoid(double x)
    {
        double result;

        result =  1 / (1 + Math.pow(Math.E,-x));

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
     * @return f(x) = (b - a) / (1 + exp(-x)) - a
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
        // initialize inner neurons
        for(int i = 0; i < mNumHiddenNeurons; i++)
        {
            mNeuronWeights[i] = getRandomDouble(WEIGHT_INIT_MIN, WEIGHT_INIT_MAX);
        }
        // initialize the output neuron
        mOutputNeuronWeight = getRandomDouble(WEIGHT_INIT_MIN, WEIGHT_INIT_MAX);
    }

    public void printNeuronWeights()
    {
        System.out.println("Current neuron weights are as follows:");
        for(int i = 0; i < mNumHiddenNeurons; i++)
        {
            System.out.format("\tInner neuron %d has weight of %5f\n",i,mNeuronWeights[i]);
        }
        // initialize the output neuron
        System.out.format("\tOutput neuron weight of %5f\n",mOutputNeuronWeight);
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
        // initialize inner neurons
        for(int i = 0; i < mNumHiddenNeurons; i++)
        {
            mNeuronWeights[i] = 0.0;
        }
        // initialize the output neuron
        mOutputNeuronWeight = 0.0;
    }

    /**
     * @param x The input vector. An array of doubles.
     * @return The value returned by the NN for this input vector
     */
    public double outputFor(double[] x)
    {
        int hiddenNeuronIndex, inputIndex, index;

        // Add the inputs to the input array
        mInputValues[BIAS_INPUT_INDEX] = 1.0;
        for(index = 1; index < x.length+1; index++)
        {
            mInputValues[index] = x[index-1];
        }

        // Calculate the outputs of the hidden neurons for the given input
        for(hiddenNeuronIndex = 0; hiddenNeuronIndex < mNumHiddenNeurons; hiddenNeuronIndex++)
        {
            mNeuronOutputs[hiddenNeuronIndex] = 0.0;

            // iterate over bias input + inputs
            for(inputIndex = 0; inputIndex < mNumInputs; inputIndex++)
            {
                mNeuronOutputs[hiddenNeuronIndex] += mNeuronWeights[hiddenNeuronIndex] * mInputValues[inputIndex];
            }

            // Apply the activation function to the weighted sum
            mNeuronOutputs[hiddenNeuronIndex] = customSigmoid(mNeuronOutputs[hiddenNeuronIndex]);
        }

        // Calculate the output of the output neuron
        mOutputValue = 0.0;
        for(hiddenNeuronIndex = 0; hiddenNeuronIndex < mNumHiddenNeurons; hiddenNeuronIndex++)
        {
            mOutputValue += mNeuronOutputs[hiddenNeuronIndex];
        }

        // Apply the activation function to the weighted sum
        mOutputValue = customSigmoid(mOutputValue);

        return mOutputValue;
    }

    public double train(double[] x, double argValue)
    {
        return 1.0;
    }

    public void save(File argFile)
    {

    }

    public void load(String argFileName) throws IOException
    {

    }
}