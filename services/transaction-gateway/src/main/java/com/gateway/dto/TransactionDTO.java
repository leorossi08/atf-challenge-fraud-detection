package com.gateway.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.PositiveOrZero;

import java.util.List;

public record TransactionDTO(
    @NotBlank
    String transactionId,
    @NotBlank
    String cardId,
    @PositiveOrZero
    double amount,
    @NotBlank
    String timestamp,
    String nome_estabelecimento,
    String tipo_estabelecimento,
    String cidade,
    String device_type,
    Boolean is_new_device,
    Double temperatura_celsius,
    String codigo_postal,
    Integer score_interno,
    String turno,
    String valor_categoria,
    String cidade_regiao,
    List<Double> features
) {
    public TransactionDTO(
            String transactionId,
            String cardId,
            double amount,
            String timestamp,
            List<Double> features) {
        this(
                transactionId,
                cardId,
                amount,
                timestamp,
                "Gas Station",
                "Fuel",
                "São Paulo",
                "mobile",
                false,
                24.0,
                "01000-000",
                500,
                "noite",
                "medio",
                "Sudeste",
                features
        );
    }
}
