import java.io.File;
import java.io.IOException;

/**
 * This interface is common to both the Neural Net and LUT interfaces.
 * The idea is that you should be able to easily switch the LUT
 * for the Neural Net since the interfaces are identical.
 * @author mmkakulp
 *
 */
public interface CommonInterface
{
    /**
     * @param x The input vector. An array of doubles.
     * @return The value returned by th LUT or NN for this input vector
     */
    double outputFor(double[] x);

    /**
     * This method will tell the NN or the LUT the output
     * value that should be mapped to the given input vector. I.e.
     * the desired correct output value for an input.
     * @param x The input vector
     * @param argValue The new value to elarn
     * @return The error in the output for that input vector
     */
    double train(double[] x, double argValue);

    /**
     * A method to write either a LUT or weights of a neural net to a file.
     * @param argFile of type File
     */
    void save(File argFile);

    /**
     * Loads the LUT or neural net weights from a file. The load must of course
     * have knowledge of how the data was written out by the save method.
     * You should raise an error in the case that an attempt is being made to
     * load data into a LUT or neural net whose structure does not match
     * the data in the file (e.g. wrong number of hidden neurons).
     * @param argFileName of type File
     * @throws IOException in case the input file does not match the current neural network type and structure
     */
    void load(String argFileName) throws IOException;
}
