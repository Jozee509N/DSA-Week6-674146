# DSA 2050 – Week 6 Guided Regression Lab

A guided supervised-regression project that predicts **sales (KES)** from business characteristics such as advertising spend, store visits, discounts, region, customer segment and product category. The notebook walks through the full workflow: data audit, leakage-safe preprocessing, baselines, model comparison, cross-validation, residual and subgroup error analysis, an overfitting demonstration and a final recommendation.

**File:** `DSA2050_Week6_Guided_Regression_Lab.ipynb`

---

## Business Question

Can available business characteristics be used to estimate sales?

- **Task:** supervised regression
- **Target:** `Sales_KES`
- **Features:** `AdvertisingSpend_KES`, `StoreVisits`, `DiscountPercent`, `Region`, `CustomerSegment`, `ProductCategory`

---

## Dataset

`DSA2050_Week6_Sales_Data.csv` (not included in this repo; you must supply it)

| Property | Value |
|---|---|
| Rows | 320 (each row is one sales record) |
| Columns | 7 |
| Numeric | `AdvertisingSpend_KES`, `StoreVisits`, `DiscountPercent`, `Sales_KES` |
| Categorical | `Region` (4 values), `CustomerSegment` (3), `ProductCategory` (4) |
| Missing values | None |
| Duplicates | None |

`Sales_KES` ranges from about 17,805 to 295,825 KES (mean about 169,252) with a roughly bell-shaped distribution and no extreme outliers.

---

## Requirements

- Python 3.9+
- pandas
- numpy
- matplotlib
- scikit-learn
- Jupyter (Notebook, JupyterLab or VS Code)

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

---

## How to Run

1. Place `DSA2050_Week6_Sales_Data.csv` somewhere accessible.
2. **Update the file path** in the data-loading cell. It currently points to a local OneDrive path on the original author's machine:
   ```python
   df = pd.read_csv("C:\\Users\\jutha\\...\\DSA2050_Week6_Sales_Data.csv")
   ```
   Change it to your own path, for example `pd.read_csv("DSA2050_Week6_Sales_Data.csv")`.
3. Run all cells from top to bottom. A fixed `RANDOM_STATE = 42` keeps results reproducible.

---

## Workflow

1. **Load and inspect data:** shape, dtypes, row meaning and target identification.
2. **Data audit:** missing values, duplicates, summary statistics and a histogram of `Sales_KES`.
3. **Task definition:** features and target separated into `X` and `y`.
4. **Train/test split:** 80/20 (256 training rows, 64 test rows).
5. **Preprocessing pipeline:** applied inside the model pipeline to avoid information leakage.
   - Numeric: median imputation, then `StandardScaler`
   - Categorical: most-frequent imputation, then `OneHotEncoder(handle_unknown="ignore")`
6. **Baselines:** `DummyRegressor` using mean and median.
7. **Models:** Linear Regression, Ridge Regression (`alpha=1.0`), and a shallow Decision Tree (`max_depth=4`, `min_samples_leaf=8`).
8. **Evaluation metrics:** MAE, RMSE and R², computed with a reusable `evaluate_regression` helper.
9. **Cross-validation:** 5-fold CV on the training data only.
10. **Residual analysis:** residual plot and actual-vs-predicted plot for the linear model.
11. **Error analysis:** the 10 largest errors, and MAE by customer segment.
12. **Overfitting check:** shallow tree versus an unconstrained deep tree.
13. **Extension:** Ridge with `alpha` of 0.1, 1, 10 and 100.
14. **Final recommendation and limitations.**

---

## Key Results

### Test-set performance

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Ridge Regression | 13,714 | 17,459 | 0.870 |
| Linear Regression | 13,758 | 17,470 | 0.870 |
| Decision Tree (depth 4) | 23,776 | 28,768 | 0.648 |
| Median Baseline | 39,147 | 48,649 | -0.006 |
| Mean Baseline | 39,197 | 48,716 | -0.008 |

### 5-fold cross-validation (training data)

| Model | CV MAE | CV RMSE | CV R² |
|---|---|---|---|
| Ridge Regression | 15,971 | 20,104 | 0.848 |
| Linear Regression | 15,977 | 20,127 | 0.848 |
| Decision Tree | 24,498 | 30,609 | 0.651 |

### Other findings

- **Residuals** are centered near zero with no obvious pattern, though a few large over-predictions exist (the worst is about -47,860 KES).
- **Largest errors** cluster in Nairobi (6 of the top 10) and the Retail segment (6 of the top 10); product categories are fairly spread out.
- **MAE by segment:** Retail 15,561 (27 rows), Corporate 14,300 (19 rows), SME 10,481 (18 rows).
- **Overfitting demo:** an unconstrained tree reaches a training MAE of 0 but a test MAE of about 26,798, showing it memorized the training data.
- **Ridge alpha sweep:** alpha 0.1 to 10 performs similarly (R² about 0.87); alpha 100 over-regularizes (R² 0.783).

---

## Conclusion

Ridge Regression is the best candidate, cutting error by roughly 65% compared with the baselines (MAE about 13.7k KES versus about 39k KES). However, the model should not be deployed immediately. Retail errors are higher than other segments, and a few large over-predictions need investigation first.

## Limitations

- **Small dataset:** 320 rows, with only 64 in the test set, so subgroup results are noisy.
- **Limited predictors:** no seasonality, competitor activity, stock levels, pricing detail or macroeconomic factors.
- **Correlation, not causation:** the model captures associations, so changing ad spend or discounts will not guarantee the predicted sales.
- **Single split:** test metrics come from one random split, so results may shift with different splits or future data drift.

## Possible Next Steps

- Add mean residual (bias) and MAPE alongside MAE for each segment.
- Use bootstrap confidence intervals for subgroup MAE.
- Test Segment × Region interaction terms or tree-based ensembles.
- Collect time, customer-history and stock data.
