package org.frontend.controllers;

import javafx.concurrent.Task;
import javafx.fxml.FXML;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressIndicator;
import javafx.scene.layout.StackPane;
import javafx.scene.paint.Color;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import org.frontend.models.PredictionResult;
import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.util.UUID;

/**
 * Controller class for handling user interactions on the main page.
 * Allows users to upload a medical image and request a prediction from the backend.
 */
public class MainController {

    @FXML
    public StackPane uploadArea;
    @FXML
    private Button predictButton;
    @FXML
    private Label fileNameLabel;
    @FXML
    private ProgressIndicator loadingIndicator;
    @FXML
    private Label resultLabel;

    private Stage stage;
    private File selectedFile;

    /**
     * Initializes the controller after its root element has been completely processed.
     * Sets event handlers for UI components.
     */
    @FXML
    private void initialize() {
        uploadArea.setOnMouseClicked(e -> selectFile());
        predictButton.setOnAction(e -> makePrediction());
    }

    /**
     * Opens a FileChooser dialog for the user to select an image file.
     * Preffered formats: png, jpg, jpeg.
     */
    private void selectFile() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Select Medical Image");

        fileChooser.getExtensionFilters().addAll(
                new FileChooser.ExtensionFilter("Image Files", "*.png", "*.jpg", "*.jpeg"),
                new FileChooser.ExtensionFilter("PNG Files", "*.png"),
                new FileChooser.ExtensionFilter("JPEG Files", "*.jpg", "*.jpeg")
        );

        selectedFile = fileChooser.showOpenDialog(stage);

        if (selectedFile != null) {
            fileNameLabel.setText(selectedFile.getName());
            fileNameLabel.setTextFill(Color.rgb(44, 62, 80));
            predictButton.setDisable(false);
            resultLabel.setVisible(false);
        }
    }

    /**
     * Sends the selected image to the backend server and displays the prediction result.
     * Uses a background task to prevent blocking the UI.
     */
    private void makePrediction() {
        if (selectedFile == null) {
            showAlert("Error", "Please select an image file first.");
            return;
        }

        loadingIndicator.setVisible(true);
        predictButton.setDisable(true);
        resultLabel.setVisible(false);

        Task<PredictionResult> predictionTask = new Task<>() {
            @Override
            protected PredictionResult call() throws Exception {
                return sendImageForPrediction(selectedFile);
            }
        };

        predictionTask.setOnSucceeded(e -> {
            displayResult(predictionTask.getValue());
            loadingIndicator.setVisible(false);
            predictButton.setDisable(false);
        });

        predictionTask.setOnFailed(e -> {
            showAlert("Error", "Failed to get prediction. Please try again.");
            loadingIndicator.setVisible(false);
            predictButton.setDisable(false);
        });

        new Thread(predictionTask).start();
    }

    /**
     * Sends a multipart POST request containing the image file to the backend.
     *
     * @param imageFile the image file to send
     * @return the parsed PredictionResult from the server response
     * @throws IOException          if there's an error with file I/O or the HTTP request
     * @throws InterruptedException if the request is interrupted
     */
    private PredictionResult sendImageForPrediction(File imageFile) throws IOException, InterruptedException {
        HttpClient client = HttpClient.newHttpClient();

        String boundary = "----Boundary" + UUID.randomUUID();
        String mimeType = Files.probeContentType(imageFile.toPath());

        // Prepare multipart part headers and footers
        String fileName = imageFile.getName();
        byte[] fileBytes = Files.readAllBytes(imageFile.toPath());

        String partHeader = "--" + boundary + "\r\n" +
                "Content-Disposition: form-data; name=\"image\"; filename=\"" + fileName + "\"\r\n" +
                "Content-Type: " + mimeType + "\r\n\r\n";

        String partFooter = "\r\n--" + boundary + "--\r\n";

        byte[] headerBytes = partHeader.getBytes();
        byte[] footerBytes = partFooter.getBytes();

        // Combine all byte arrays
        byte[] requestBody = new byte[headerBytes.length + fileBytes.length + footerBytes.length];
        System.arraycopy(headerBytes, 0, requestBody, 0, headerBytes.length);
        System.arraycopy(fileBytes, 0, requestBody, headerBytes.length, fileBytes.length);
        System.arraycopy(footerBytes, 0, requestBody, headerBytes.length + fileBytes.length, footerBytes.length);

        // Create HTTP request
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("http://localhost:8080/api/predict"))
                .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                .POST(HttpRequest.BodyPublishers.ofByteArray(requestBody))
                .build();

        // Send request and get response
        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() == 200) {
            JSONObject json = new JSONObject(response.body());
            String label = json.getString("label");
            double confidence = json.getDouble("confidence");
            return new PredictionResult(label, confidence);
        } else {
            throw new IOException("Server error: " + response.statusCode() + " - " + response.body());
        }
    }

    /**
     * Displays the prediction result on the UI with colored text based on label.
     *
     * @param result the prediction result to display
     */
    private void displayResult(PredictionResult result) {
        String text = String.format("%s (confidence - %.1f%%)",
                result.label().toUpperCase(),
                result.accuracy() * 100);

        resultLabel.setText(text);
        resultLabel.setVisible(true);
    }

    /**
     * Displays an error alert with the specified title and message.
     *
     * @param title   the alert dialog title
     * @param message the error message
     */
    private void showAlert(String title, String message) {
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }
}
