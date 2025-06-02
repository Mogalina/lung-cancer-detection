package org.frontend.models;

/**
 * Represents the result of a prediction from the backend service.
 *
 * @param label    The label returned by the prediction.
 *                 For example: "benign", "malign", or "normal".
 * @param accuracy The accuracy/confidence level of the prediction, from 0.0 to 1.0.
 */
public record PredictionResult(String label, double accuracy) {

    /**
     * Constructs a new {@code PredictionResult} with the given label and accuracy.
     *
     * @param label    the predicted label (e.g., "malign")
     * @param accuracy the prediction confidence, from 0.0 to 1.0
     */
    public PredictionResult {
    }

    /**
     * Returns the label predicted by the model.
     *
     * @return the prediction label
     */
    @Override
    public String label() {
        return label;
    }

    /**
     * Returns the confidence level of the prediction.
     *
     * @return the accuracy as a double between 0.0 and 1.0
     */
    @Override
    public double accuracy() {
        return accuracy;
    }
}

