package org.frontend;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.io.IOException;
import java.util.Objects;

/**
 * Entry point for the JavaFX application.
 * This class sets up the main window (Stage) and loads the FXML user interface.
 */
public class HelloApplication extends Application {

    /**
     * Starts the JavaFX application.
     * Sets up the stage, loads the FXML layout, applies styles, and displays the main window.
     *
     * @param primaryStage the primary window of the application
     * @throws IOException if loading the FXML file fails
     */
    @Override
    public void start(Stage primaryStage) throws IOException {
        primaryStage.setTitle("Lung Cancer Detection");
        primaryStage.setResizable(false);

        // Load the main FXML layout
        FXMLLoader loader = new FXMLLoader(getClass().getResource("/org/frontend/pages/main-page.fxml"));

        // Create the scene using the loaded layout with specified dimensions
        Scene scene = new Scene(loader.load(), 450, 580);

        // Apply external stylesheet
        scene.getStylesheets().add(Objects.requireNonNull(getClass().getResource("/org/frontend/styles/main-page.css")).toExternalForm());

        // Set the scene on the primary stage
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    /**
     * Main method that launches the JavaFX application.
     *
     * @param args command-line arguments (not used)
     */
    public static void main(String[] args) {
        launch();
    }
}
