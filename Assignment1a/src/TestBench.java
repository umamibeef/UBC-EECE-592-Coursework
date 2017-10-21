
public class TestBench implements NeuralNet
{
    public static void main(String[] args)
    {
        NeuralNet NeuralNetObj = new NeuralNet(2, 4, 0.5, 0.5, 0, 1);

        System.out.format("Hi. Neural net instantiated with %i inputs and %i hidden neurons.", NeuralNetObj.mNumInputs, NeuralNetObj.mNumHiddenNeurons);
    }
}

