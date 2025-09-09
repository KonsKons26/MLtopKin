import os

import numpy as np

import pandas as pd

from collections import Counter

from datetime import datetime

import json

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    matthews_corrcoef, accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier

import optuna
from optuna.samplers import TPESampler

from mrmr import mrmr_classif


def time():
    """Returns the current time in a formatted string."""
    now = datetime.now()
    return f"{now:%Y/%m/%d-%H:%M:%S}"


def specificity_score(y_true, y_pred):
    """Calculate specificity score."""
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


VALID_MODELS = {
    "LogisticRegression": LogisticRegression(),
    "GaussianNB": GaussianNB(),
    "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
    "SVC": SVC(),
    "RandomForestClassifier": RandomForestClassifier(),
    "LGBMClassifier": LGBMClassifier(verbosity=-1)
}

VALID_METRICS = {
    "mcc": matthews_corrcoef,
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "roc_auc": roc_auc_score,
    "specificity": specificity_score
}


class NestedCV:
    """
    Parameters
    ----------
    model_name : str
        Name of the model to use. Must be one of the keys in VALID_MODELS.
    X : pd.DataFrame
        Features for the model.
    y : pd.Series
        Target variable for the model.
    models_dir : str
        Directory to save the trained models.
    results_dir : str
        Directory to save the results.
    n_rounds : int, optional (default=10)
        Number of rounds for the nested cross-validation.
    n_outers : int, optional (default=5)
        Number of outer folds for the nested cross-validation.
    n_inners : int, optional (default=5)
        Number of inner folds for the nested cross-validation.
    n_optuna_trials : int, optional (default=100)
        Number of trials for hyperparameter optimization with Optuna.
    metric : str, optional (default="mcc")
        Metric to optimize during hyperparameter tuning. Must be one of the
        keys in VALID_METRICS.
    mrmr : bool, optional (default=True)
        Whether to use mrmr for feature selection
    n_features : in, optional (default=10)
        Number of features to select
    random_state_base : int, optional (default=42)
        Base random state for reproducibility. This will be used to generate
        different random states for each fold.
    """

    def __init__(
            self,
            model_name: str,
            X: pd.DataFrame,
            y: pd.Series,
            models_dir: str,
            results_dir: str,
            n_rounds: int = 10,
            n_outers: int = 5,
            n_inners: int = 5,
            n_optuna_trials: int = 100,
            metric: str = "mcc",
            optuna_direction: str = "maximize",
            mrmr: bool = True,
            n_features: int =None,
            random_state_base: int = 42
        ):
        """Initialize the NestedCV class."""
        self.model_name = model_name
        self.model = VALID_MODELS[model_name]
        self.X = X
        self.y = y
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.n_rounds = n_rounds
        self.n_outers = n_outers
        self.n_inners = n_inners
        self.n_optuna_trials = n_optuna_trials
        self.metric_name = metric
        self.metric = VALID_METRICS[metric]
        self.optuna_direction = optuna_direction
        self.mrmr = mrmr
        self.n_features = n_features
        self.random_state_base = random_state_base

        self.hyperparam_spaces = self._define_hyperparameter_spaces()
        self.results = {
            r: {
                o_f: {} for o_f in range(n_outers)
            } for r in range(n_rounds)
        }


    def _define_hyperparameter_spaces(self):
        """
        Define the hyperparameter spaces for each classifier.

        This method returns a dictionary where the keys are classifier
        names and the values are functions that take an Optuna trial
        object and return a dictionary of hyperparameters.
        The hyperparameters are defined using Optuna's suggest methods.

        Returns
        -------
        dict
            A dictionary containing the hyperparameter spaces for each
            classifier.
        """
        return {
            "LogisticRegression": lambda trial: {
                "solver": "saga",
                "penalty": "elasticnet",
                "C": trial.suggest_float("C", 1e-5, 1e5, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
                "max_iter": trial.suggest_int("max_iter", 100, 10_000),
                "class_weight": trial.suggest_categorical(
                    "class_weight", ["balanced", None]
                )
            },
            "GaussianNB": lambda trial: {
                "var_smoothing": trial.suggest_float(
                    "var_smoothing", 1e-10, 1e-5, log=True
                )
            },
            "LinearDiscriminantAnalysis": lambda trial: {
                "solver": trial.suggest_categorical(
                    "solver", ["svd", "lsqr", "eigen"]
                ),
                "shrinkage": trial.suggest_categorical(
                    "shrinkage", [None, "auto"]
                ) if not trial.params.get("solver") == "svd" else None,
                "priors": trial.suggest_categorical(
                    "priors", [None, [0.5, 0.5]]
                )
            },
            "SVC": lambda trial: {
                "C": trial.suggest_float("C", 0.1, 1, log=True),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                "coef0": trial.suggest_float("coef0", 0.0, 1.0),
                "kernel": trial.suggest_categorical(
                    "kernel", ["linear", "poly", "rbf", "sigmoid"]
                ),
                "degree": trial.suggest_int(
                    "degree", 1, 3
                ) if trial.params.get("kernel") == "poly" else 3,
                "class_weight": trial.suggest_categorical(
                    "class_weight", ["balanced", None]
                )
            },
            "RandomForestClassifier": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 10, 100),
                "criterion": trial.suggest_categorical(
                    "criterion", ["gini", "entropy"]
                ),
                "max_depth": None,
                "min_samples_split": trial.suggest_int(
                    "min_samples_split", 2, 20
                ),
                "random_state": 42
            },
            "LGBMClassifier": lambda trial: {
                "boosting_type": trial.suggest_categorical(
                    "boosting_type", ["gbdt", "dart", "rf"]
                ),
                "num_leaves": trial.suggest_int("num_leaves", 10, 100),
                "max_depth": trial.suggest_int("max_depth", 1, 100),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-5, 1.0, log=True
                ),
                "n_estimators": trial.suggest_int("n_estimators", 10, 1_000),
                "objective": "binary",
                "min_child_samples": trial.suggest_int(
                    "min_child_samples", 1, 100
                ),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
                "verbose": -1,
                "random_state": 42,
                # For rf
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                "bagging_fraction": trial.suggest_float(
                    "bagging_fraction", 0.1, 0.99
                ),
                "feature_fraction": trial.suggest_float(
                    "feature_fraction", 0.1, 0.99
                ),
            }
        }


    def _calculate_metrics(
            self,
            y_true: pd.Series,
            y_pred: pd.Series
        ):
        return {
            "mcc": matthews_corrcoef(y_true, y_pred),
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_pred),
            "specificity": specificity_score(y_true, y_pred)
        }


    def _objective(
            self,
            trial: optuna.Trial,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            objective_random_state: int
        ):
        """
        Objective function for Optuna hyperparameter optimization.

        Parameters
        ----------
        trial : optuna.Trial
            The Optuna trial object.
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training target variable.
        objective_random_state : int
            Random state for reproducibility within this objective's inner
            folds.

        Returns
        -------
        float
            The mean score of the model based on the specified metric.
        """
        hyperparams = self.hyperparam_spaces[self.model_name](trial)
        model = VALID_MODELS[self.model_name].__class__(**hyperparams)

        skf = StratifiedKFold(
            n_splits=self.n_inners,
            shuffle=True,
            random_state=objective_random_state
        )

        scores = []
        trial_selected_features_list = []

        # --- TUNING --- #
        for train_idx, val_idx in skf.split(X_train, y_train):
            # Split the data for the current inner fold
            X_tr = X_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_val = y_train.iloc[val_idx]

            # Scaling the data for the current inner fold
            inner_scaler = StandardScaler()
            X_tr_scaled = pd.DataFrame(
                inner_scaler.fit_transform(X_tr),
                columns=X_tr.columns,
                index = X_tr.index
            )
            X_val_scaled = pd.DataFrame(
                inner_scaler.transform(X_val),
                columns=X_val.columns,
                index = X_val.index
            )

            # Feature selection for the current inner fold
            current_selected_features = X_tr_scaled.columns.tolist()
            if self.mrmr:
                current_selected_features = mrmr_classif(
                    X=X_tr_scaled,
                    y=y_tr,
                    K=self.n_features,
                    show_progress=False
                )
                X_tr_scaled = X_tr_scaled[current_selected_features]
                X_val_scaled = X_val_scaled[current_selected_features]
            # TODO: possibly add more fs methods

            # Accumulate features selected in this inner fold
            trial_selected_features_list.extend(current_selected_features)

            # Fit the model and evaluate
            model.fit(X_tr_scaled, y_tr)

            if hasattr(model, 'classes_') and isinstance(model.classes_, list):
                model.classes_ = np.array(model.classes_)

            y_pred = model.predict(X_val_scaled)
            score = self.metric(y_val, y_pred)
            scores.append(score)

            # Handle Optuna pruning
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # After all inner folds for this trial, store the aggregated feature
        # counts
        trial.set_user_attr(
            "selected_features_in_trial_counts",
            Counter(trial_selected_features_list)
        )

        return np.mean(scores)


    def run_nested_cv(self, verbose: bool = True):
        """
        Run nested cross-validation with hyperparameter tuning.

        This method performs nested cross-validation, where the outer loop
        is used for model evaluation and the inner loop is used for
        hyperparameter tuning using Optuna.
        It identifies the 'n_features_to_select' most commonly selected features
        from the best Optuna trial's inner folds for the final model in the
        outer loop. It also tracks feature selection counts for the best trial
        in each outer fold and saves results and trained models.
        """

        # Ensure output directories exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # --- ROUNDS --- #

        for round_idx in range(self.n_rounds):
            if verbose:
                print(
                    f"[{time()}]"
                    f" Starting Round {round_idx + 1}/{self.n_rounds}..."
                )

            round_random_state = self.random_state_base + round_idx

            # --- OUTER CROSS-VALIDATION LOOP --- #

            outer_skf = StratifiedKFold(
                n_splits=self.n_outers,
                shuffle=True,
                random_state=round_random_state
            )

            for outer_fold_idx, (outer_train_idx, outer_val_idx) in enumerate(
                outer_skf.split(self.X, self.y)
            ):
                if verbose:
                    print(
                        f"[{time()}]"
                        f"\tOuter Fold {outer_fold_idx + 1}/{self.n_outers}..."
                    )

                # Split data into outer train and validation sets (unscaled)
                outer_X_train = self.X.iloc[outer_train_idx]
                outer_y_train = self.y.iloc[outer_train_idx]
                outer_X_val = self.X.iloc[outer_val_idx]
                outer_y_val = self.y.iloc[outer_val_idx]

                # Scale the outer training and validation data
                outer_scaler = StandardScaler()
                outer_X_train_scaled = pd.DataFrame(
                    outer_scaler.fit_transform(outer_X_train),
                    columns=outer_X_train.columns,
                    index=outer_X_train.index
                )
                outer_X_val_scaled = pd.DataFrame(
                    outer_scaler.transform(outer_X_val),
                    columns=outer_X_val.columns,
                    index=outer_X_val.index
                )

                # --- INNER LOOP --- HYPERPARAMETER TUNING WITH OPTUNA --- #
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                # Optuna study for hyperparameter optimization
                study = optuna.create_study(
                    sampler=TPESampler(seed=round_random_state),
                    direction=self.optuna_direction,
                    study_name=(
                        f"{self.model_name}_round_{round_idx}"
                        f"_outer_fold_{outer_fold_idx}"
                    )
                )
                # The inner loop actually happens inside the _objective ;)
                study.optimize(
                    lambda trial: self._objective(
                        trial,
                        outer_X_train,  # Will be scaled in there
                        outer_y_train,
                        round_random_state
                    ),
                    n_trials=self.n_optuna_trials,
                    show_progress_bar=True,
                    n_jobs=-1
                )

                # Retrieve best parameters and the full feature selection counts
                # from the best trial
                best_params = study.best_params
                feature_selection_counts_for_best_trial = study.best_trial.user_attrs.get(
                    "selected_features_in_trial_counts", Counter()
                )


                # --- RESUME ON THE OUTER FOLD --- #

                most_common_features_list = [
                    feature for feature, _
                    in feature_selection_counts_for_best_trial.most_common(
                        self.n_features
                    )
                ]

                # If no feature selection method is specified or no
                # features were selected by mRMR, use all features from
                # the scaled dataset.
                if not self.mrmr or not most_common_features_list:
                    final_features_for_outer_fold = outer_X_train_scaled.columns.tolist()
                    if verbose:
                        print(
                            f"[{time()}]"
                            "\tNo feature selection or no features selected. "
                            f"Using all {len(final_features_for_outer_fold)}"
                            " features."
                        )
                else:
                    final_features_for_outer_fold = most_common_features_list
                    if verbose:
                        print(
                            f"[{time()}]"
                            f"\tSelected top {len(final_features_for_outer_fold)} "
                            "most common features for final model."
                        )

                # Subset the scaled outer training and validation sets with the
                # final features
                outer_X_train_final = outer_X_train_scaled[
                    final_features_for_outer_fold
                ]
                outer_X_val_final = outer_X_val_scaled[
                    final_features_for_outer_fold
                ]

                # Train the final model for this outer fold with the best
                # hyperparameters
                final_model = self.model.set_params(**best_params)
                final_model.fit(outer_X_train_final, outer_y_train)

                # Evaluate the final model on the outer validation set
                outer_y_pred = final_model.predict(outer_X_val_final)
                outer_metrics = self._calculate_metrics(
                    outer_y_val, outer_y_pred
                )

                # Store Results for the Current Outer Fold
                self.results[round_idx][outer_fold_idx] = {
                    "metrics": outer_metrics,
                    "hyperparams": best_params,
                    "feature_selection_counts": dict(
                        feature_selection_counts_for_best_trial
                    ),
                    "features_used_for_final_model": final_features_for_outer_fold
                }
                if verbose:
                    print(
                        f"[{time()}]"
                        f"\tOuter Fold {outer_fold_idx + 1} completed."
                        f" Main Metric ({self.metric_name}):"
                        f"\t{outer_metrics[self.metric_name]:.4f}"
                    )


        # --- ROUNDS ENDED --- #

        # After all rounds and folds are complete, save the entire results
        # dictionary
        results_filename = f"{self.model_name}_nested_cv_results_{datetime.now(
        ):%Y%m%d-%H%M%S}.json"
        results_filepath = os.path.join(self.results_dir, results_filename)
        with open(results_filepath, "w") as f:
            serializable_results = convert_numpy_types(self.results)
            # always indent=4 for pretty jsons <3
            json.dump(serializable_results, f, indent=4)
        print(f"[{time()}] All Nested CV results saved to {results_filepath}")


    def summarize_results(self):
        """
        Generates a comprehensive summary DataFrame from the nested
        cross-validation results.

        Returns
        -------
        pd.DataFrame
            A DataFrame where each row represents an outer fold, containing its
            metrics, hyperparameters, feature selection counts, and the
            features used for the final model.

        collections.Counter
            A Counter object summarizing the global feature selection
            frequencies across all outer folds and their respective best trials.
        """

        if not self.results:
            print("[WARN] No results found. Run 'run_nested_cv' first.")
            return pd.DataFrame(), Counter()

        all_fold_records = []
        for round_idx, outer_folds_data in self.results.items():
            for outer_fold_idx, fold_data in outer_folds_data.items():
                record = {
                    "round": round_idx,
                    "outer_fold": outer_fold_idx,
                    **fold_data["metrics"],
                    "hyperparams": fold_data["hyperparams"],
                    "feature_selection_counts": fold_data[
                        "feature_selection_counts"
                    ],
                    "features_used_for_final_model": fold_data[
                        "features_used_for_final_model"
                    ]
                }
                all_fold_records.append(record)

        summary_df = pd.DataFrame(all_fold_records)

        # Calculate global feature selection frequencies
        global_feature_selection_counts = Counter()
        for counts_dict in summary_df["feature_selection_counts"]:
            global_feature_selection_counts.update(counts_dict)

        print(f"\n--- Summary of Nested CV Results ({self.model_name}) ---")
        print(f"Primary Metric ({self.metric_name}):")
        print(summary_df[self.metric_name].agg(
            ["mean", "median", "std"]).to_string())

        print("\n--- Round-wise Metrics (Median) ---")
        # Display median of primary metric per round
        print(summary_df.groupby(
            "round")[self.metric_name].median().to_string())

        if self.mrmr:
            print("\n--- Top 10 Global Feature Selection Frequencies ---")
            print(convert_numpy_types(
                global_feature_selection_counts.most_common(self.n_features)
            ))

        return summary_df, global_feature_selection_counts


def complete_pipeline(
        model_name: str,
        X: pd.DataFrame,
        y: pd.DataFrame,
        models_dir: str,
        results_dir: str,
        n_rounds: int = 5,
        n_outers: int = 5,
        n_inners: int = 10,
        n_optuna_trials: int = 50,
        metric: str = "mcc",
        optuna_direction: str = "maximize",
        mrmr: bool = True,
        n_features: int = 10,
        random_state_base=42
    ):
    """
    Executes the complete nested cross-validation pipeline, including model
    training, hyperparameter tuning, feature selection, and results
    summarization, saving both detailed and summarized results.

    Parameters
    ----------
    model_name : str
        Name of the model to use. Must be one of the keys in VALID_MODELS.
    X : pd.DataFrame
        Features for the model.
    y : pd.Series
        Target variable for the model.
    models_dir : str
        Directory to save the trained models.
    results_dir : str
        Directory to save the results.
    n_rounds : int, optional (default=2)
        Number of rounds for the nested cross-validation.
    n_outers : int, optional (default=2)
        Number of outer folds for the nested cross-validation.
    n_inners : int, optional (default=2)
        Number of inner folds for the nested cross-validation.
    n_optuna_trials : int, optional (default=5)
        Number of trials for hyperparameter optimization with Optuna.
    metric : str, optional (default="mcc")
        Metric to optimize during hyperparameter tuning. Must be one of the
        keys in VALID_METRICS.
    optuna_direction : str, optional (default="maximize")
        Direction for Optuna optimization ("minimize" or "maximize").
    mrmr : bool, optional (default=True)
        Whether to use mrmr for feature selection.
    n_features : int, optional (default=10)
        Number of features to select if mrmr is True.
    random_state_base : int, optional (default=42)
        Base random state for reproducibility.
    """
    # --- RUN NCV --- #
    print(f"[{time()}] Starting Nested CV for {model_name}...")
    nested_cv = NestedCV(
        model_name=model_name,
        X=X,
        y=y,
        models_dir=models_dir,
        results_dir=results_dir,
        n_rounds=n_rounds,
        n_outers=n_outers,
        n_inners=n_inners,
        n_optuna_trials=n_optuna_trials,
        metric=metric,
        optuna_direction=optuna_direction,
        mrmr=mrmr,
        n_features=n_features,
        random_state_base=random_state_base
    )
    nested_cv.run_nested_cv()

    # --- CREATE SUMMARY --- #
    print(f"[{time()}] Summarizing results...")
    summary_df, global_feature_counts = nested_cv.summarize_results()

    # Save summary results to CSV
    summary_filename = f"{model_name}_summary_results_{datetime.now(
    ):%Y%m%d-%H%M%S}.csv"
    summary_filepath = os.path.join(results_dir, summary_filename)

    temp_summary_df = summary_df.copy()

    temp_summary_df["hyperparams"] = temp_summary_df[
        "hyperparams"
    ].apply(convert_numpy_types).apply(json.dumps)
    temp_summary_df["feature_selection_counts"] = temp_summary_df[
        "feature_selection_counts"
    ].apply(convert_numpy_types).apply(json.dumps)
    temp_summary_df["features_used_for_final_model"] = temp_summary_df[
        "features_used_for_final_model"
    ].apply(convert_numpy_types).apply(json.dumps)

    temp_summary_df.to_csv(summary_filepath, index=False)
    print(f"[{time()}] Summary results saved to {summary_filepath}")


def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {
            convert_numpy_types(k): convert_numpy_types(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [convert_numpy_types(elem) for elem in obj]
    elif isinstance(obj, np.integer): # Catches int64, int32, etc.
        return int(obj)
    elif isinstance(obj, np.floating): # Catches float64, float32, etc.
        return float(obj)
    elif isinstance(obj, np.bool_): # Catches numpy booleans
        return bool(obj)
    else:
        return obj


if __name__ == "__main__":

    # ------------------------------------------------------------------------ #
    # --- EXAMPLE WITH BREAST CANCER DATASET --- #
    # Has MCC between 0.87 and 0.94 with:
    # n_rounds=1
    # n_outers=5
    # n_inners=5
    # n_optuna_trials=10
    # ------------------------------------------------------------------------ #

    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer(as_frame=True)
    X_dummy = data["data"]
    y_dummy = data["target"]
    # Define directories
    models_dir = "trained_models_final"
    results_dir = "cv_results_final"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    complete_pipeline(
        "LogisticRegression",
        X_dummy,
        y_dummy,
        models_dir,
        results_dir,
        n_rounds=1, # Reduced for quick testing
        n_outers=5, # Reduced for quick testing
        n_inners=5, # Reduced for quick testing
        n_optuna_trials=10 # Reduced for quick testing
    )
