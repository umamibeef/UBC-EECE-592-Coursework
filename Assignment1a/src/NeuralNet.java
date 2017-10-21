import java.lang.Math;

public class NeuralNet implements NeuralNetInterface
{
    private double mArgA;
    private double mArgB;

    /**
     * Constructor for NeuralNet
     * @param argNumInputs The number of inputs in your input vector
     * @param argNumHidden The number of hidden neurons in your hidden layer. Only a single hidden layer is supported
     * @param argLearningRate The learning rate coefficient
     * @param argMomentumTerm The momentum coefficient
     * @param argA Integer lower bound of sigmoid used by the output neuron only.
     * @param argB Integer upper bound of sigmoid used by the output neuron only.
    */
    public abstract NeuralNet(int argNumInputs,
                              int argNumHidden,
                              double argLearningRate,
                              double argMomentumTerm,
                              double argA,
                              double argB)
    {
        // update our private variables
        mArgA = argA;
        mArgB = argB;
    }

    /**
     * This methods implements the sigmoid function
     * @param x The input
     * @return f(x) 1 / (1 + exp(-x))
     */
    double sigmoid(double x)
    {
        return 1 / (1 + Math.pow(Math.E,-x));
    }

    /**
     * This method implements a general sigmoid with asymptotes bounded by (a,b)
     * @param x The input
     * @return f(x) = (b - a) / (1 + exp(-x)) - a
     */
    double customSigmoid(double x)
    {
        return (mArgB - mArgA) * signmoid(x) + mArgA;
    }

    void initializeWeights()
    {

    }

    void zeroWeights()
    {

    }

    double outputFor(double[] x);
    {

    }

    double train(double[] x, double argValue)
    {

    }

    void save(File argFile)
    {

    }

    void load(String argFileName) throws IOException
    {

    }
}
