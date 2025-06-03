package org.service;

import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;

import java.io.IOException;

/**
 * Service responsible for sending image data to an external prediction model via HTTP requests using a reactive
 * {@link WebClient}.
 */
@Service
public class PredictionService {

    private final WebClient webClient;

    /**
     * Constructs a PredictionService using a {@link WebClient.Builder}.
     * The WebClient is configured with a base URL pointing to the prediction model service.
     *
     * @param webClientBuilder the builder used to create the WebClient instance
     */
    public PredictionService(WebClient.Builder webClientBuilder) {
        this.webClient = webClientBuilder.baseUrl("http://localhost:6000").build();
    }

    /**
     * Sends an image to the prediction model endpoint and returns the model's response as a String.
     *
     * @param image the multipart image file to be sent
     * @return the prediction result returned by the model as a String
     * @throws IOException if an I/O error occurs reading the image bytes
     */
    public String sendToModel(MultipartFile image) throws IOException {
        return webClient.post()
                .uri("/predict")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(BodyInserters.fromMultipartData("file", new ByteArrayResource(image.getBytes()) {
                    @Override
                    public String getFilename() {
                        return image.getOriginalFilename();
                    }
                }))
                .retrieve()
                .bodyToMono(String.class)
                .block();
    }
}
