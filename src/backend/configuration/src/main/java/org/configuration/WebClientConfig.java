package org.configuration;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.WebClient;

/**
 * Configuration class to provide a {@link WebClient.Builder} bean for the application.
 *
 * <p>
 * The {@link WebClient.Builder} is used to create and customize {@link WebClient} instances, which are reactive,
 * non-blocking HTTP clients for making HTTP requests.
 * </p>
 *
 * <p>
 * By defining this bean, it allows for centralized configuration and easier injection of WebClient builders wherever
 * needed in the Spring context.
 * </p>
 */
@Configuration
public class WebClientConfig {

    /**
     * Provides a {@link WebClient.Builder} bean.
     *
     * @return a new instance of {@link WebClient.Builder} to build WebClient instances
     */
    @Bean
    public WebClient.Builder webClientBuilder() {
        return WebClient.builder();
    }
}
