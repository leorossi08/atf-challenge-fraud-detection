package com.gateway.model;

import com.gateway.dto.TransactionDTO;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.Id;
import jakarta.persistence.PrePersist;
import jakarta.persistence.PreUpdate;
import jakarta.persistence.Table;

import java.time.Instant;

@Entity
@Table(name = "transactions")
public class TransactionRecord {

    @Id
    @Column(name = "transaction_id", nullable = false, updatable = false, length = 100)
    private String transactionId;

    @Column(name = "card_id", nullable = false, length = 100)
    private String cardId;

    @Column(nullable = false)
    private double amount;

    @Column(name = "transaction_timestamp", nullable = false, length = 80)
    private String transactionTimestamp;

    @Column(name = "merchant_name", length = 100)
    private String merchantName;

    @Column(name = "merchant_type", length = 80)
    private String merchantType;

    @Column(name = "city", length = 80)
    private String city;

    @Column(name = "region", length = 60)
    private String region;

    @Column(name = "device_type", length = 40)
    private String deviceType;

    @Column(name = "is_new_device")
    private Boolean isNewDevice;

    @Column(name = "shift", length = 40)
    private String shift;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 20)
    private TransactionStatus status;

    @Column(name = "is_fraud")
    private Boolean isFraud;

    @Column(name = "fraud_probability")
    private Double fraudProbability;

    @Column(name = "status_message", length = 500)
    private String statusMessage;

    @Column(name = "created_at", nullable = false, updatable = false)
    private Instant createdAt;

    @Column(name = "updated_at", nullable = false)
    private Instant updatedAt;

    protected TransactionRecord() {
    }

    public static TransactionRecord queued(TransactionDTO transaction) {
        TransactionRecord record = new TransactionRecord();
        record.transactionId = transaction.transactionId();
        record.cardId = transaction.cardId();
        record.amount = transaction.amount();
        record.transactionTimestamp = transaction.timestamp();
        record.merchantName = transaction.nome_estabelecimento() != null ? transaction.nome_estabelecimento() : "Online Shop 1";
        record.merchantType = transaction.tipo_estabelecimento() != null ? transaction.tipo_estabelecimento() : "E-commerce";
        record.city = transaction.cidade() != null ? transaction.cidade() : "São Paulo";
        record.region = transaction.cidade_regiao() != null ? transaction.cidade_regiao() : "Sudeste";
        record.deviceType = transaction.device_type() != null ? transaction.device_type() : "mobile";
        record.isNewDevice = Boolean.TRUE.equals(transaction.is_new_device());
        record.shift = transaction.turno() != null ? transaction.turno() : "noite";
        record.status = TransactionStatus.QUEUED;
        return record;
    }

    @PrePersist
    void onCreate() {
        Instant now = Instant.now();
        createdAt = now;
        updatedAt = now;
    }

    @PreUpdate
    void onUpdate() {
        updatedAt = Instant.now();
    }

    public void markProcessing() {
        status = TransactionStatus.PROCESSING;
        statusMessage = null;
    }

    public void complete(boolean fraud, double probability) {
        status = TransactionStatus.COMPLETED;
        isFraud = fraud;
        fraudProbability = probability;
        statusMessage = null;
    }

    public void fail(String message) {
        status = TransactionStatus.FAILED;
        statusMessage = message;
    }

    public String getTransactionId() {
        return transactionId;
    }

    public String getCardId() {
        return cardId;
    }

    public double getAmount() {
        return amount;
    }

    public String getTransactionTimestamp() {
        return transactionTimestamp;
    }

    public String getMerchantName() {
        return merchantName;
    }

    public String getMerchantType() {
        return merchantType;
    }

    public String getCity() {
        return city;
    }

    public String getRegion() {
        return region;
    }

    public String getDeviceType() {
        return deviceType;
    }

    public Boolean getIsNewDevice() {
        return isNewDevice;
    }

    public String getShift() {
        return shift;
    }

    public TransactionStatus getStatus() {
        return status;
    }

    public Boolean getIsFraud() {
        return isFraud;
    }

    public Double getFraudProbability() {
        return fraudProbability;
    }

    public String getStatusMessage() {
        return statusMessage;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public Instant getUpdatedAt() {
        return updatedAt;
    }
}
