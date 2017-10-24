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

    // Learning rate and momentum term
    public double mLearningRate;
    public double mMomentumTerm;

    // Array to store input values to the neural network, first index is bias input of 1
    public double[] mInputValues = new double[MAX_INPUTS];

    // Array to store neuron weights of hidden layer
    public static double[][] mHiddenNeuronWeights = new double[MAX_HIDDEN_NEURONS][MAX_INPUTS];
    // Array to store previous weights
    public static double[][] mPreviousHiddenNeuronWeights = new double[MAX_HIDDEN_NEURONS][MAX_INPUTS];
    // Array to store neuron outputs of the hidden layer
    public double[] mHiddenNeuronOutputs = new double[MAX_HIDDEN_NEURONS];
    // Array to store neuron errors of the hidden layer
    public double[] mHiddenNeuronErrors = new double[MAX_HIDDEN_NEURONS];

    // Array to store the output neuron's input weights
    public static double[] mOutputNeuronWeights = new double[MAX_HIDDEN_NEURONS];
    // Array to store the previous output neuron's weights
    public static double[] mPreviousOutputNeuronWeights = new double[MAX_HIDDEN_NEURONS];
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
                              double argB)
    {
        // Update our private variables
        mArgA = argA;
        mArgB = argB;
        // Add one here so that we don't worry about it later in the code (for bias)
        mNumInputs = argNumInputs+1;
        mNumHiddenNeurons = argNumHidden+1;
        // Record the learning and momentum rates
        mLearningRate = argLearningRate;
        mMomentumTerm = argMomentumTerm;

        // Zero out the weights (also clears previous entry)
        zeroWeights();

        // Update the bias value to one
        mInputValues[BIAS_INPUT_INDEX] = 1.0;

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
                mHiddenNeuronWeights[i][j] = getRandomDouble(WEIGHT_INIT_MIN, WEIGHT_INIT_MAX);
            }
            // initialize the output neuron
            mOutputNeuronWeights[i] = getRandomDouble(WEIGHT_INIT_MIN, WEIGHT_INIT_MAX);
        }

        mHiddenNeuronWeights[0][0] = 42.0;
        mHiddenNeuronWeights[0][1] = 42.0;
        mHiddenNeuronWeights[0][2] = 42.0;
    }

    /**
     * Updates the weights based on the current backpropagated error
     */
    public void updateWeights()
    {
        int i, j;
        double momentumTerm, learningTerm;

        for(i = 1; i < mNumHiddenNeurons; i++)
        {
            // Update the weights to the output neuron
            // Calculate terms
            momentumTerm = mMomentumTerm * (mOutputNeuronWeights[i] - mPreviousOutputNeuronWeights[i]);
            learningTerm = mLearningRate * mOutputNeuronError * mHiddenNeuronOutputs[i];
            // Save old values
            mPreviousOutputNeuronWeights = mOutputNeuronWeights.clone();
            mOutputNeuronWeights[i] = mOutputNeuronWeights[i] + momentumTerm + learningTerm;

            // No input weights for bias, continue
            if(i != 0)
            {
                for(j = 0; j < mNumInputs; j++)
                {
                    // Update the weights to the hidden neurons
                    // Calculate terms
                    momentumTerm = mMomentumTerm * (mHiddenNeuronWeights[i][j] - mPreviousHiddenNeuronWeights[i][j]);
                    learningTerm = mLearningRate * mHiddenNeuronErrors[i] * mInputValues[j];
                    // Save old values
                    mPreviousHiddenNeuronWeights = mHiddenNeuronWeights.clone();
                    // Update hidden neuron weights
                    mHiddenNeuronWeights[i][j] = mHiddenNeuronWeights[i][j] + momentumTerm + learningTerm;
                }
            }
        }
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

    public void printNeuronWeights()
    {
        int i, j;

        System.out.println("Current neuron weights are as follows:");
        System.out.println("\tInput neuron weights");
        for(i = 0; i < mNumHiddenNeurons; i++)
        {
            for (j = 0; j < mNumInputs; j++)
            {
//                System.out.format("\t\tInner neuron %d input %d has weight of %5f\n", i, j, mHiddenNeuronWeights[i][j]);
                System.out.format("%d %d %5f\n", i, j, mHiddenNeuronWeights[i][j]);
            }
        }
        System.out.println("\tOutput Neuron Weights");
        for(i = 0; i < mNumHiddenNeurons; i++)
        {
//            System.out.format("\t\tOutput neuron input %d has weight of %5f\n", i, mOutputNeuronWeights[i]);
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
                mPreviousHiddenNeuronWeights[i][j] = 0.0;
                mHiddenNeuronWeights[i][j] = 0.0;
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
        int i, j;

//        System.out.format("Input values: %1f %1f %1f\n", mInputValues[0], mInputValues[1], mInputValues[2]);

        // Calculate the outputs of the hidden neurons for the given input
        // We use the mHiddenNeuronOutputs[0] as the bias input to the output stage
        mHiddenNeuronOutputs[BIAS_INPUT_INDEX] = 1.0;
        for(i = 1; i < mNumHiddenNeurons; i++)
        {
            mHiddenNeuronOutputs[i] = 0.0;

            // iterate over bias input + inputs
            for(j = 0; j < mNumInputs; j++)
            {
                mHiddenNeuronOutputs[i] += mHiddenNeuronWeights[i][j] * mInputValues[j];
            }

            // Apply the activation function to the weighted sum
            mHiddenNeuronOutputs[i] = customSigmoid(mHiddenNeuronOutputs[i]);
        }

        // Calculate the output of the output neuron
        mOutputNeuronValue = 0.0;
        for(i = 0; i < mNumHiddenNeurons; i++)
        {
            mOutputNeuronValue += mHiddenNeuronOutputs[i] * mOutputNeuronWeights[i];
        }

        // Apply the activation function to the weighted sum
        mOutputNeuronValue = customSigmoid(mOutputNeuronValue);

        return mOutputNeuronValue;
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
        int i;

        // calculate the output value
        outputFor(x);

        // calculate the output error
        mOutputNeuronError = (argValue - mOutputNeuronValue) * customSigmoidDerivative(mOutputNeuronValue);

        // backpropagate the output error
        for(i = 0; i < mNumHiddenNeurons; i++)
        {
            mHiddenNeuronErrors[i] = mOutputNeuronError * mOutputNeuronWeights[i] * customSigmoidDerivative(mHiddenNeuronOutputs[i]);
        }

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