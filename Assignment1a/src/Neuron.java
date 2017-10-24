public class Neuron
{
    static final int MAXIMUM_INPUTS = 10;
    int mNumInputs;

    double[] mNeuronWeights = new double[MAXIMUM_INPUTS];

    public Neuron(int numInputs)
    {
        mNumInputs = numInputs;
    }
}