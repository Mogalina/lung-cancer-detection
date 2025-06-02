package org.controller;

import org.service.PredictionService;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

/**
 * REST controller that exposes an endpoint for sending images to a prediction model.
 */
@RestController
@RequestMapping("/api")
public class PredictionController {

    private final PredictionService predictionService;

    /**
     * Constructs a PredictionController with the given PredictionService.
     *
     * @param predictionService the service that handles sending images to the prediction model
     */
    public PredictionController(PredictionService predictionService) {
        this.predictionService = predictionService;
    }

    /**
     * Handles POST requests to "/api/predict" with a multipart image file.
     *
     * @param image the uploaded image file sent as a multipart request parameter named "image"
     * @return a ResponseEntity containing the prediction result as a String if successful, or an error message with
     *      HTTP 500 status if an exception occurs
     */
    @PostMapping("/predict")
    public ResponseEntity<String> predict(@RequestParam("image") MultipartFile image) {
        try {
            String result = predictionService.sendToModel(image);
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("Error during prediction: " + e.getMessage());
        }
    }
}
