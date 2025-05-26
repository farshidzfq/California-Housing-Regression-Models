import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing


def load_data():
    """Load California Housing dataset as a DataFrame."""
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    print("California Housing dataset loaded successfully.")
    print(df.head())
    print(df.info())
    print(df.describe())
    return df


def preprocess_data(df, target_column):
    """Split features and target, and prepare preprocessing pipeline."""
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    numerical_features = X.columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), numerical_features)],
        remainder='passthrough'
    )
    return X, y, preprocessor


def train_models(X_train, y_train, preprocessor):
    """Train different regression models and return pipelines."""
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100, n_jobs=-1)
    }

    pipelines = {}
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', model)])
        pipeline.fit(X_train, y_train)
        pipelines[name] = pipeline
        print(f"{name} model trained.")
    return pipelines


def evaluate_models(pipelines, X_test, y_test):
    """Evaluate each model pipeline and return metrics."""
    results = {}
    for name, pipeline in pipelines.items():
        y_pred = pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
        print(f"\n{name} Evaluation:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R2 Score: {r2:.4f}")
    return results


def optimize_random_forest(X_train, y_train, preprocessor):
    """Optimize Random Forest model using GridSearchCV."""
    param_grid = {
        'regressor__n_estimators': [50, 100],
        'regressor__max_depth': [10, 20, None],
        'regressor__min_samples_split': [2, 5]
    }
    pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))])

    grid_search = GridSearchCV(pipeline_rf, param_grid, cv=3,
                               scoring='neg_mean_squared_error',
                               verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"\nBest Random Forest parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


def plot_feature_importance(pipeline, feature_names):
    """Plot feature importance for tree-based models."""
    regressor = pipeline.named_steps['regressor']
    if hasattr(regressor, 'feature_importances_'):
        importances = regressor.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
        plt.title(f'Feature Importance - {type(regressor).__name__}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
    else:
        print("Model does not have feature_importances_ attribute.")


def plot_actual_vs_predicted(pipeline, X_test, y_test, model_name):
    """Plot Actual vs Predicted values for regression results."""
    y_pred = pipeline.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='w', linewidth=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Line')
    plt.xlabel('Actual Median House Value ($100k)')
    plt.ylabel(f'Predicted Median House Value ({model_name} - $100k)')
    plt.title(f'Actual vs. Predicted Prices - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def predict_new_sample(pipeline, feature_names, sample_features):
    """Predict house price for a new sample."""
    df_sample = pd.DataFrame([sample_features], columns=feature_names)
    pred = pipeline.predict(df_sample)[0]
    price_dollars = pred * 100000
    print("\nNew Sample Prediction:")
    print(df_sample)
    print(f"Predicted Median House Value: ${price_dollars:,.2f} (Raw: {pred:.4f} in $100k units)")


def main():
    # Load and prepare data
    df = load_data()
    target = 'MedHouseVal'
    X, y, preprocessor = preprocess_data(df, target)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTrain samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # Train models
    pipelines = train_models(X_train, y_train, preprocessor)

    # Evaluate models
    results = evaluate_models(pipelines, X_test, y_test)

    # Select best model based on RMSE
    best_model_name = min(results, key=lambda k: results[k]['RMSE'])
    best_pipeline = pipelines[best_model_name]
    print(f"\nBest model by RMSE: {best_model_name} (RMSE={results[best_model_name]['RMSE']:.4f})")

    # Optimize Random Forest if not already best
    if best_model_name != 'Random Forest':
        optimized_rf = optimize_random_forest(X_train, y_train, preprocessor)
        y_pred_opt = optimized_rf.predict(X_test)
        rmse_opt = np.sqrt(mean_squared_error(y_test, y_pred_opt))
        print(f"Optimized Random Forest RMSE: {rmse_opt:.4f}")

        if rmse_opt < results[best_model_name]['RMSE']:
            best_model_name = 'Optimized Random Forest'
            best_pipeline = optimized_rf
            print(f"Updated best model to: {best_model_name}")

    # Visualizations
    plot_feature_importance(best_pipeline, X.columns.tolist())
    plot_actual_vs_predicted(best_pipeline, X_test, y_test, best_model_name)

    # Predict new sample
    new_sample = {
        'MedInc': 3.5,   # Median income in block (~$35,000)
        'HouseAge': 30,
        'AveRooms': 5.5,
        'AveBedrms': 1.1,
        'Population': 1200,
        'AveOccup': 2.8,
        'Latitude': 34.0,
        'Longitude': -118.0
    }
    predict_new_sample(best_pipeline, X.columns.tolist(), new_sample)


if __name__ == "__main__":
    main()
