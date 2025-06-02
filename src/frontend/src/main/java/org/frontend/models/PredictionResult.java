package org.frontend.models;
/**
 * Represents the result of a prediction from the backend service.
 */
public class PredictionResult {

    /**
     * The label returned by the prediction.
     * For example: "benign", "malign", or "normal".
     */
    private final String label;

    /**
     * The accuracy/confidence level of the prediction, from 0.0 to 1.0.
     */
    private final double accuracy;

    /**
     * Constructs a new {@code PredictionResult} with the given label and accuracy.
     *
     * @param label    the predicted label (e.g., "malign")
     * @param accuracy the prediction confidence, from 0.0 to 1.0
     */
    public PredictionResult(String label, double accuracy) {
        this.label = label;
        this.accuracy = accuracy;
    }

    /**
     * Returns the label predicted by the model.
     *
     * @return the prediction label
     */
    public String getLabel() {
        return label;
    }

    /**
     * Returns the confidence level of the prediction.
     *
     * @return the accuracy as a double between 0.0 and 1.0
     */
    public double getAccuracy() {
        return accuracy;
    }
}

