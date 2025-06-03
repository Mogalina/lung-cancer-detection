package org.configuration;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

/**
 * Configuration class to set up CORS (Cross-Origin Resource Sharing) settings for the application.
 *
 * <p>
 * This class defines a {@link WebMvcConfigurer} bean to customize CORS mappings globally.
 * CORS allows or restricts resources to be requested from another domain outside the domain from which
 * the resource originated.
 * </p>
 *
 * <p>
 * The configuration here allows all origins, supports common HTTP methods, accepts all headers,
 * and enables credentials with a cache duration of 3600 seconds for preflight requests.
 * </p>
 */
@Configuration
public class CorsConfig {

    /**
     * Creates a {@link WebMvcConfigurer} bean that customizes CORS settings.
     *
     * @return a WebMvcConfigurer instance with overridden addCorsMappings method
     */
    @Bean
    public WebMvcConfigurer corsConfigurer() {
        return new WebMvcConfigurer() {

            /**
             * Configures CORS mappings for the application.
             *
             * @param registry the CorsRegistry to add the CORS configuration to
             */
            @Override
            public void addCorsMappings(CorsRegistry registry) {
                registry.addMapping("/**")
                        // Allow requests from any origin
                        .allowedOrigins("*")
                        // Allow common HTTP methods
                        .allowedMethods("GET", "POST", "PUT", "DELETE", "OPTIONS")
                        // Allow all headers
                        .allowedHeaders("*")
                        // Restrict credentials such as cookies and authorization headers
                        .allowCredentials(false);
            }
        };
    }
}
