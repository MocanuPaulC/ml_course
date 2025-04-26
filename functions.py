import matplotlib.pyplot as plt
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

import numpy as np
import optuna
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve

def plot_data_distributions(dataframe):
    """
    Plot data distributions for each column in the DataFrame.

    For numeric columns, it plots a histogram.
    For categorical columns, it plots a bar chart of value counts.
    """
    # Determine the number of rows needed for subplots
    num_cols = len(dataframe.columns)
    num_rows = num_cols // 2 if num_cols % 2 == 0 else (num_cols // 2) + 1

    fig, axs = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))

    axs = axs.ravel() if num_cols > 1 else [axs]

    # Iterate over each column to plot
    for idx, column in enumerate(dataframe.columns):
        # Check data type of column
        if pd.api.types.is_numeric_dtype(dataframe[column]):
            # Plot histogram for numeric columns
            axs[idx].hist(dataframe[column].dropna(), bins=15, edgecolor='black', alpha=0.7)
            axs[idx].set_title(f'Histogram of {column}')
        else:
            # Plot bar chart for categorical columns
            value_counts = dataframe[column].value_counts()
            axs[idx].bar(value_counts.index, value_counts.values, color='skyblue', alpha=0.7)
            axs[idx].set_title(f'Bar Chart of {column}')

            # Set the tick positions and labels
            axs[idx].set_xticks(range(len(value_counts)))
            axs[idx].set_xticklabels(value_counts.index, rotation=45, ha='right')

        axs[idx].set_xlabel(column)
        axs[idx].set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()



# Custom Preprocessor to handle your specific data cleaning
class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.job_education_mode = None

    def fit(self, X, y=None):
        #  job -> education mapping
        strong_jobs = ['JobCat11', 'JobCat6', 'JobCat7', 'JobCat5', 'JobCat9']
        self.job_education_mode = X[X['job'].isin(strong_jobs)].groupby('job')['education'].agg(
            lambda x: x.mode().iloc[0])
        return self

    def transform(self, X):
        dataset = X.copy()

        # Drop columns
        dataset = dataset.drop(columns=['duration', 'poutcome'])

        # Fill missing 'contact' with mode
        dataset['contact'] = dataset['contact'].fillna(dataset['contact'].mode()[0])

        # Drop rows where 'job' is missing
        dataset = dataset.dropna(subset=['job'])

        # Fill missing 'education' based on job-specific mode, fallback to 'secondary'
        def fill_education(row):
            if pd.isnull(row['education']):
                if row['job'] in self.job_education_mode:
                    return self.job_education_mode[row['job']]
                else:
                    return 'secondary'
            else:
                return row['education']

        dataset['education'] = dataset.apply(fill_education, axis=1)

        return dataset



def prepare_data(df):
    #  categorical and numerical features
    categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month']
    numerical_features = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']

    cleaner = CustomPreprocessor()
    df_clean = cleaner.fit_transform(df)

    # Separate target after cleaning
    y = df_clean['y'].copy()
    X = df_clean.drop(columns=['y'])

    # transformers
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(sparse_output=False, drop='first')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # pipeline
    feature_pipeline = Pipeline(steps=[
        ('feature_preprocessor', preprocessor)
    ])


    X = feature_pipeline.fit_transform(X)

    # Encode y
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return X, y, feature_pipeline, label_encoder


import optuna
import xgboost as xgb
import lightgbm as lgb

import catboost as cb

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score


def train_test_cv_models(X, y):
    """
    Trains, tunes, and tests multiple models for classification.
    """
    results = {}

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Apply SMOTE
    sm = SMOTETomek(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
    X_train, y_train = X_resampled, y_resampled

    # Define all models to be tuned
    tuned_models = {
        'knn': KNeighborsClassifier(),
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'random_forest': RandomForestClassifier(random_state=42),
        'naive_bayes': GaussianNB(),
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'xgboost': xgb.XGBClassifier(tree_method='hist', eval_metric='logloss', random_state=42),
        'lightgbm': lgb.LGBMClassifier(random_state=42),
        'catboost': cb.CatBoostClassifier(task_type='CPU', verbose=0, random_seed=42),
    }

    for name, model in tuned_models.items():
        print(f"Tuning and training {name}...")

        def objective(trial):
            if name == 'knn':
                params = {
                    'n_neighbors': trial.suggest_int('n_neighbors', 3, 20),
                    'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                    'p': trial.suggest_int('p', 1, 2),
                }
                model_obj = KNeighborsClassifier(**params)

            elif name == 'decision_tree':
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                }
                model_obj = DecisionTreeClassifier(random_state=42, class_weight='balanced', **params)

            elif name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 600),
                    'max_depth': trial.suggest_int('max_depth', 3, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                }
                model_obj = RandomForestClassifier(random_state=42, class_weight='balanced', **params)

            elif name == 'naive_bayes':
                model_obj = GaussianNB()

            elif name == 'logistic_regression':
                penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
                params = {
                    'C': trial.suggest_float('C', 0.001, 10.0),
                    'penalty': penalty,
                    'solver': 'saga',
                }
                if penalty == 'elasticnet':
                    params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)

                model_obj = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, **params)

            elif name == 'xgboost':
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 600),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                }
                scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
                model_obj = xgb.XGBClassifier(tree_method='hist', eval_metric='logloss', random_state=42,
                                              scale_pos_weight=scale_pos_weight, **params)

            elif name == 'lightgbm':
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 600),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                }
                model_obj = lgb.LGBMClassifier(random_state=42, class_weight='balanced', verbosity=-1, **params)

            elif name == 'catboost':
                params = {
                    'depth': trial.suggest_int('depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'iterations': trial.suggest_int('iterations', 100, 600),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                    'border_count': trial.suggest_int('border_count', 32, 255),
                    'task_type': 'CPU',
                    'verbose': 0,
                    'random_seed': 42,
                }
                class_weights = [1, (np.sum(y_train == 0) / np.sum(y_train == 1))]
                params['class_weights'] = class_weights
                model_obj = cb.CatBoostClassifier(**params)

            # Cross-validation on training set, score using F1
            score = cross_val_score(model_obj, X_train, y_train, scoring='f1', cv=5)
            return np.mean(score)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)  # Increased from 20 to 50

        best_params = study.best_params

        # Rebuild model with best params
        if name == 'knn':
            model = KNeighborsClassifier(**best_params)
        elif name == 'decision_tree':
            model = DecisionTreeClassifier(random_state=42, class_weight='balanced', **best_params)
        elif name == 'random_forest':
            model = RandomForestClassifier(random_state=42, class_weight='balanced', **best_params)
        elif name == 'naive_bayes':
            model = GaussianNB()
        elif name == 'logistic_regression':
            model = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                solver='saga',  # FORCE saga again manually
                **best_params
            )
        elif name == 'xgboost':
            model = xgb.XGBClassifier(tree_method='hist', eval_metric='logloss', random_state=42, **best_params)
        elif name == 'lightgbm':
            model = lgb.LGBMClassifier(random_state=42, class_weight='balanced', verbosity=-1, **best_params)
        elif name == 'catboost':
            model = cb.CatBoostClassifier(task_type='CPU', verbose=0, random_seed=42, **best_params)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        if probs is not None:
            precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
            f1_scores = 2 * (precisions * recalls) / (
                        precisions + recalls + 1e-8)  # small epsilon to avoid division by zero
            best_threshold_idx = f1_scores.argmax()
            best_threshold = thresholds[best_threshold_idx]

            print(f"Best threshold for {name}: {best_threshold:.4f}")

            # Use best threshold to get new final predictions
            preds = (probs >= best_threshold).astype(int)

            # Then compute metrics again
            acc = accuracy_score(y_test, preds)
            precision = precision_score(y_test, preds)
            recall = recall_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            roc_auc = roc_auc_score(y_test, probs)  # still use probs for ROC-AUC
        else:
            acc = accuracy_score(y_test, preds)
            precision = precision_score(y_test, preds)
            recall = recall_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            roc_auc = None

        results[name] = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'model': model,
            'best_params': best_params,
            'best_threshold': best_threshold,
        }

    return results


def compare_models(results):
    """
    Plots Accuracy, Precision, Recall, F1 Score, and ROC-AUC for each model.
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    model_names = list(results.keys())

    # Build data arrays
    metric_values = {metric: [results[model].get(metric, 0) for model in model_names] for metric in metrics}

    x = np.arange(len(model_names))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots(figsize=(16, 7))

    # Plot bars for each metric
    rects1 = ax.bar(x - 2 * width, metric_values['accuracy'], width, label='Accuracy')
    rects2 = ax.bar(x - width, metric_values['precision'], width, label='Precision')
    rects3 = ax.bar(x, metric_values['recall'], width, label='Recall')
    rects4 = ax.bar(x + width, metric_values['f1'], width, label='F1 Score')
    rects5 = ax.bar(x + 2 * width, metric_values['roc_auc'], width, label='ROC-AUC')

    # Add labels, title, legend
    ax.set_ylabel('Scores')
    ax.set_xlabel('Models')
    ax.set_title('Model Comparison (Accuracy, Precision, Recall, F1, ROC-AUC)')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.legend()

    # Add value labels on top
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    for rects in [rects1, rects2, rects3, rects4, rects5]:
        autolabel(rects)

    fig.tight_layout()
    plt.show()

