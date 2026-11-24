package com.gateway.dto;

import com.gateway.model.TransactionRecord;
import com.gateway.model.TransactionStatus;

import java.time.Instant;

public record TransactionStatusResponse(
        String transactionId,
        String cardId,
        double amount,
        String timestamp,
        String nome_estabelecimento,
        String tipo_estabelecimento,
        String cidade,
        String cidade_regiao,
        String device_type,
        Boolean is_new_device,
        String turno,
        TransactionStatus status,
        Boolean isFraud,
        Double fraudProbability,
        String message,
        Instant createdAt,
        Instant updatedAt
) {
    public TransactionStatusResponse(
            String transactionId,
            String cardId,
            double amount,
            String timestamp,
            TransactionStatus status,
            Boolean isFraud,
            Double fraudProbability,
            String message,
            Instant createdAt,
            Instant updatedAt
    ) {
        this(
                transactionId,
                cardId,
                amount,
                timestamp,
                "Gas Station",
                "Fuel",
                "São Paulo",
                "Sudeste",
                "mobile",
                false,
                "noite",
                status,
                isFraud,
                fraudProbability,
                message,
                createdAt,
                updatedAt
        );
    }

    public static TransactionStatusResponse from(TransactionRecord record) {
        return new TransactionStatusResponse(
                record.getTransactionId(),
                record.getCardId(),
                record.getAmount(),
                record.getTransactionTimestamp(),
                record.getMerchantName(),
                record.getMerchantType(),
                record.getCity(),
                record.getRegion(),
                record.getDeviceType(),
                record.getIsNewDevice(),
                record.getShift(),
                record.getStatus(),
                record.getIsFraud(),
                record.getFraudProbability(),
                record.getStatusMessage(),
                record.getCreatedAt(),
                record.getUpdatedAt()
        );
    }
}
