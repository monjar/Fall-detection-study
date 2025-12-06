# Ensemble Fall Detector - Test Results

## Dataset

- Total sequences: 500
- Falls: 219
- No Falls: 281
- Sequence length: 64

## Ensemble Configuration

- Voting strategy: weighted
- Detectors: physics, temporal, geometry
- Weights:
  - physics: 0.30
  - temporal: 0.25
  - geometry: 0.20
  - neural: 0.15
  - anomaly: 0.10

## Results Summary

| Detector | Accuracy | Precision | Recall | F1 | FAR |
|----------|----------|-----------|--------|-------|-----|
| ensemble | 73.80% | 0.6507 | 0.8676 | 0.7436 | 36.30% |
| physics | 73.40% | 0.6311 | 0.9452 | 0.7569 | 43.06% |
| geometry | 70.80% | 0.7110 | 0.5616 | 0.6276 | 17.79% |
| temporal | 48.60% | 0.4500 | 0.7808 | 0.5710 | 74.38% |

## Key Insights

- **Best Accuracy**: ensemble (73.80%)
- **Best F1 Score**: physics (0.7569)
- **Lowest False Alarm Rate**: geometry (17.79%)

## Confusion Matrices

### ensemble

```
                Predicted
              No Fall    Fall
Actual No Fall  179      102    
       Fall     29       190    
```

### physics

```
                Predicted
              No Fall    Fall
Actual No Fall  160      121    
       Fall     12       207    
```

### geometry

```
                Predicted
              No Fall    Fall
Actual No Fall  231      50     
       Fall     96       123    
```

### temporal

```
                Predicted
              No Fall    Fall
Actual No Fall  72       209    
       Fall     48       171    
```

