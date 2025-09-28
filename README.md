# 🫀 Heart Failure Mortality Risk Prediction

A machine learning model to predict heart failure mortality risk using clinical features, deployed as a FastAPI service with comprehensive AWS deployment options.

## 🎯 Project Overview

This project implements a **recall-optimized** logistic regression model to identify high-risk heart failure patients, prioritizing early detection over precision to ensure no critical cases are missed.

### Key Features
- **Recall-first approach**: Tuned threshold (0.2095) achieves 100% recall on test data
- **Production-ready**: FastAPI service with comprehensive error handling
- **AWS deployment**: Multiple deployment options (Lambda, Elastic Beanstalk, ECS, EC2)
- **Model governance**: Complete model card and configuration tracking

## 🧠 Methodology (CRISP-DM)

### Phase 1 — Business & Data Understanding

**Goal**: Prioritize care by catching high-risk patients (Recall-first approach).

**Metrics**: 
- AUPRC (class imbalance handling)
- Recall, Precision, F1, AUROC
- Brier score for calibration assessment

**EDA**: Schema analysis, missingness patterns, target balance, feature distributions, and key clinical relationships.

### Phase 2 — Data Preparation

**Pipeline (ColumnTransformer)**:
- **Numerics**: Median impute → Yeo–Johnson transform → RobustScaler
- **Binary flags**: Mode impute; maintain 0/1 encoding
- **Leakage control**: Drop `time` feature to prevent data leakage

### Phase 3 — Modeling

**Model Selection**:
- Baselines → **Logistic Regression** (L2, class_weight='balanced'), Decision Tree, Random Forest
- 5-fold stratified out-of-fold (OOF) cross-validation for honest model comparison
- Consistent preprocessing pipeline across all folds

### Phase 4 — Evaluation

**Model Selection**: Best model chosen by CV AUPRC performance

**Threshold Tuning**: 
- Tuned to achieve Recall ≈ 0.80 using OOF PR curve
- Verified performance on held-out test set

**Comprehensive Evaluation**:
- ROC/PR curves for both default and tuned thresholds
- Confusion matrices comparison
- Calibration assessment (Brier score + reliability plots)
- **Fairness analysis**: Performance slices by sex and age bands
- **Cost analysis**: Trade-off exploration (example: 5×FN + 1×FP cost structure)

### Phase 5 — Deployment

**Artifacts**:
- Single scikit-learn pipeline export (`.joblib`)
- Threshold configuration in `config.json`
- FastAPI service with `/predict` endpoint returning probabilities & labels
- Comprehensive model card for governance & operations

## 📊 Model Performance

| Threshold | AUROC | AUPRC | Recall | Precision | F1 | Brier |
|-----------|-------|-------|--------|-----------|----|----|
| Default (0.5) | 0.550 | 0.105 | 0.000 | 0.000 | 0.000 | 0.201 |
| **Tuned (0.2095)** | **0.550** | **0.105** | **1.000** | **0.100** | **0.182** | **0.201** |

## 🏗️ Architecture

```
├── heart_failure_logreg_pipeline.joblib  # Trained model pipeline
├── config.json                           # Model configuration & threshold
├── serve.py                              # FastAPI application
├── model_card.md                         # Model documentation
├── requirements.txt                      # Python dependencies
├── Dockerfile                            # Container configuration
├── AWS_DEPLOYMENT_GUIDE.md               # Deployment instructions
└── deploy-aws.sh                         # Automated deployment script
```

## 🚀 Quick Start

### Local Development

1. **Clone and setup**:
   ```bash
   git clone https://github.com/ArshanBhanage/Assignment-1.git
   cd Assignment-1
   pip install -r requirements.txt
   ```

2. **Run the API**:
   ```bash
   uvicorn serve:app --reload
   ```

3. **Test the endpoint**:
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "instances": [{
         "anaemia": 0, "diabetes": 1, "high_blood_pressure": 1,
         "sex": 1, "smoking": 0, "age": 65,
         "creatinine_phosphokinase": 582, "ejection_fraction": 20,
         "platelets": 265000, "serum_creatinine": 1.9, "serum_sodium": 130
       }]
     }'
   ```

### AWS Deployment

Choose your preferred deployment method:

```bash
# Automated deployment script
./deploy-aws.sh

# Or manual deployment options:
# 1. Elastic Beanstalk (Recommended)
eb init heart-failure-api --platform python-3.11
eb create heart-failure-env --instance-type t3.micro

# 2. Lambda (Serverless)
serverless deploy

# 3. Docker + ECS
docker build -t heart-failure-api .
# Follow ECS deployment guide
```

## 📋 API Documentation

### Endpoints

- **GET `/ping`**: Health check
- **POST `/predict`**: Risk prediction

### Request Format
```json
{
  "instances": [{
    "anaemia": 0,           // 0 or 1
    "diabetes": 1,          // 0 or 1  
    "high_blood_pressure": 1, // 0 or 1
    "sex": 1,               // 0 or 1
    "smoking": 0,           // 0 or 1
    "age": 65.0,            // 0-120
    "creatinine_phosphokinase": 582.0, // >= 0
    "ejection_fraction": 20.0,         // 0-100
    "platelets": 265000.0,             // >= 0
    "serum_creatinine": 1.9,           // >= 0
    "serum_sodium": 130.0              // 100-200
  }]
}
```

### Response Format
```json
{
  "probabilities": [0.45],
  "labels": [1],
  "threshold": 0.2095,
  "order": ["anaemia", "diabetes", ...]
}
```

## 💰 Deployment Costs

| Platform | Monthly Cost | Best For |
|----------|-------------|----------|
| **AWS Lambda** | $0-5 | Low traffic, development |
| **Elastic Beanstalk** | $10-20 | Quick deployment, medium traffic |
| **ECS Fargate** | $15-30 | Production, high availability |
| **EC2** | $8-15 | Cost optimization, full control |

## 🔒 Responsible AI & Ethics

### Fairness Considerations
- **Subgroup Analysis**: Monitor performance across sex and age demographics
- **Bias Mitigation**: Consider group-specific thresholds if recall gaps exceed 10-15 points
- **Regular Auditing**: Implement continuous fairness monitoring

### Model Limitations
- **Scope**: Tabular clinical snapshot; not a survival/time-to-event model
- **Data Quality**: Performance degrades with unit mismatches or excessive missingness
- **Generalization**: Trained on mock data; requires validation on real clinical data

### Operational Monitoring
- **Retrain Cadence**: Monthly or when data drift detected (PSI > 0.2)
- **Performance Tracking**: Monitor recall at operating point, FP/FN rates
- **Calibration**: Track Brier score; consider recalibration if degradation occurs

## 📈 Model Governance

### Artifacts Tracking
- **Model Version**: v1.0 (LogisticRegression + class_weight balancing)
- **Training Data**: 200 samples, 8% positive rate
- **Feature Engineering**: Median/mode imputation + Yeo-Johnson + RobustScaler
- **Threshold Tuning**: 5-fold CV optimization for recall ≈ 0.80

### Change Management
- All model changes documented in `model_card.md`
- Configuration versioning in `config.json`
- Automated testing pipeline for model updates

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For questions or support:
- Create an issue in this repository
- Review the [AWS Deployment Guide](AWS_DEPLOYMENT_GUIDE.md)
- Check the [Model Card](model_card.md) for technical details

---

**⚠️ Medical Disclaimer**: This model is for educational/research purposes only. Not intended for actual clinical decision-making without proper validation and regulatory approval.
