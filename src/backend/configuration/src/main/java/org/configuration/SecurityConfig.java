package org.configuration;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.web.SecurityFilterChain;

/**
 * SecurityConfig is a Spring configuration class that sets up web security for the application.
 *
 * <p>
 * It uses Spring Security to configure HTTP security settings, including CSRF protection and request authorization.
 * </p>
 *
 * @Configuration indicates that this class contains Spring-managed beans.
 * @EnableWebSecurity enables Spring Security's web security support.
 */
@Configuration
@EnableWebSecurity
public class SecurityConfig {

    /**
     * Defines the security filter chain bean that configures HTTP security for incoming requests.
     *
     * @param http the HttpSecurity object to configure web based security for specific HTTP requests.
     * @return the configured SecurityFilterChain.
     * @throws Exception if an error occurs during configuration.
     *
     * <p>
     * This method configures:
     * - CSRF protection to be disabled.
     * - Authorization so that any HTTP request is permitted without authentication.
     * </p>
     */
    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
                .csrf(AbstractHttpConfigurer::disable)
                .authorizeHttpRequests(authz -> authz
                        .anyRequest().permitAll()
                );

        return http.build();
    }
}
