import os
import numpy as np
import pandas as pd

import mlflow

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import imblearn

import sklearn.metrics as skl_metrics

def __generate_gene_list_with_lowest_cov_quantile(X_cv, q):
    # Calculate the CoV
    gene_expr_cov_list = 100 * (X_cv.std(ddof=1, axis=0) / X_cv.mean(axis=0))

    # Calculate the 0.25 quantile
    min_cov_threshold = np.quantile(gene_expr_cov_list, q=[q])[0]

    # Get a list of genes to removed
    low_cov_gene_list = gene_expr_cov_list.loc[(gene_expr_cov_list < min_cov_threshold)].index.tolist()

    return low_cov_gene_list

def __remove_lowest_quantile_of_genes_based_on_cov(X_train, X_test, q):
    # Remove the lowest quartile of genes based on CoV
    low_cov_gene_list = __generate_gene_list_with_lowest_cov_quantile(X_train, q=q)

    # Remove the lowest quantile of genes based on CoV
    return X_train.drop(low_cov_gene_list, axis=1), X_test.drop(low_cov_gene_list, axis=1)

def __smote_oversample(X, y):
    smote_obj = imblearn.over_sampling.SMOTE(sampling_strategy='auto',
                                             random_state=110,
                                             k_neighbors=5,
                                             n_jobs=None)

    X_smote, y_smote = smote_obj.fit_sample(X, y)

    return X_smote, y_smote

def __calculate_model_accuracy(clf, X, y):
    # Calculate accuracy on non-oversampled data
    y_pred = pd.Series(clf.predict(X), index=y.index)

    return (y_pred == y).mean()


def __calculate_sens_and_spec(y_true, y_pred):
    # Generate a classification report dict
    classification_report_dict = skl_metrics.classification_report(y_true,
                                                                   y_pred,
                                                                   output_dict=True)
    # Get the sensitivity and specificity from the report dict
    sensitivity = classification_report_dict['1']['recall']
    specificity = classification_report_dict['0']['recall']

    return sensitivity, specificity

def __split_kfold_data(X, y, train_iloc_list, test_iloc_list):
    # Create current fold train data
    X_train = X.iloc[train_iloc_list,]
    y_train = y.iloc[train_iloc_list,]

    # Create current fold train data
    X_test = X.iloc[test_iloc_list,]
    y_test = y.iloc[test_iloc_list,]

    return X_train, X_test, y_train, y_test

def __generate_cv_metrics(clf, X_cv, y_cv, param_dict):
    cv_train_accuracy_list = []
    cv_test_accuracy_list = []
    cv_test_accuracy_of_0_cases_list = []
    y_cv_test_list = []
    y_cv_test_pred_prob_list = []

    kfold_obj = KFold(n_splits=5, shuffle=True, random_state=110)

    for curr_train_iloc_list, curr_test_iloc_list in kfold_obj.split(X=X_cv, y=y_cv):

        # Get the train/test data for this fold
        curr_fold_X_train, \
        curr_fold_X_test, \
        curr_fold_y_train, \
        curr_fold_y_test = __split_kfold_data(X_cv, y_cv, curr_train_iloc_list, curr_test_iloc_list)

        # Run SMOTE on the X,y data
        if param_dict['use_smote']:
            curr_fold_X_train, curr_fold_y_train = __smote_oversample(curr_fold_X_train, curr_fold_y_train)

        # Train the model for this k-fold
        clf.fit(curr_fold_X_train, curr_fold_y_train)

        # Calculate accuracy on non-oversampled data
        cv_train_accuracy_list.append(__calculate_model_accuracy(clf, curr_fold_X_train, curr_fold_y_train))
        cv_test_accuracy_list.append(__calculate_model_accuracy(clf, curr_fold_X_test, curr_fold_y_test))

        # Calculate the test accuracy of target=0 cases
        curr_fold_y_test_pred = pd.Series(clf.predict(curr_fold_X_test), index=curr_fold_y_test.index)
        cv_test_accuracy_of_0_cases_list.append(np.mean(curr_fold_y_test_pred[(curr_fold_y_test == 0)] == 0))

        # Calculate the prediction probability
        y_cv_test_list.extend(curr_fold_y_test.tolist())
        y_cv_test_pred_prob_list.extend(clf.predict_proba(curr_fold_X_test)[:, 1])

    # Calculate AUC of the CV test folds
    cv_test_auc = skl_metrics.roc_auc_score(y_true=y_cv_test_list,
                                            y_score=y_cv_test_pred_prob_list)

    return cv_train_accuracy_list, cv_test_accuracy_list, cv_test_accuracy_of_0_cases_list, cv_test_auc

def run_mlflow_exp_V10(X_cv, X_test, y_cv, y_test, curr_params_dict, experiment_base_path):

    # Set the experiment name
    mlflow.set_tracking_uri(f'file://{experiment_base_path}')

    # Start a mlflow run
    with mlflow.start_run() as mlflow_run:
        mlflow.set_tag('description', "")

        # Define the model
        log_regr_clf = LogisticRegression(penalty='l1',
                                          C=curr_params_dict['C'],
                                          class_weight=curr_params_dict['class_weight'],
                                          random_state=110,
                                          solver=curr_params_dict['solver'],
                                          max_iter=curr_params_dict['max_iter'],
                                          verbose=0,
                                          n_jobs=None,
                                          l1_ratio=None)
        # Remove the lowest q quantile of genes based on their CoV
        X_cv, X_test = __remove_lowest_quantile_of_genes_based_on_cov(X_train=X_cv,
                                                                      X_test=X_test,
                                                                      q=curr_params_dict['lower_quantile_removed_CoV'])

        # Run custom CV so I can train on over-sampled data but test on original data
        cv_train_accuracy_list,\
        cv_test_accuracy_list,\
        cv_test_accuracy_of_0_cases_list,\
        cv_test_auc = __generate_cv_metrics(log_regr_clf, X_cv, y_cv, curr_params_dict)

        # Run SMOTE on the full X_cv,y_cv data
        if curr_params_dict['use_smote']:
            X_cv, y_cv = __smote_oversample(X_cv, y_cv)

        # Train the model on the full CV data
        log_regr_clf.fit(X_cv, y_cv)

        y_test_pred = log_regr_clf.predict(X_test)

        # Log parameters
        mlflow.log_param("log10_C", np.log10(curr_params_dict['C']))
        mlflow.log_param("log10_max_iter", np.log10(curr_params_dict['max_iter']))
        mlflow.log_param("solver", curr_params_dict['solver'])
        mlflow.log_param("class_weight", curr_params_dict['class_weight'])
        mlflow.log_param("lower_quantile_removed_CoV", curr_params_dict['lower_quantile_removed_CoV'])
        mlflow.log_param("use_smote", curr_params_dict['use_smote'])

        # Log the CV metrics
        mlflow.log_metric("cv_training_accuracy", np.mean(cv_train_accuracy_list))
        mlflow.log_metric("cv_test_accuracy", np.mean(cv_test_accuracy_list))
        mlflow.log_metric("cv_test_0_class_accuracy", np.mean(cv_test_accuracy_of_0_cases_list))

        # Log test metrics
        mlflow.log_metric("train_accuracy", __calculate_model_accuracy(log_regr_clf, X_cv, y_cv))
        mlflow.log_metric("test_accuracy", __calculate_model_accuracy(log_regr_clf, X_test, y_test))
        mlflow.log_metric("train_auc", skl_metrics.roc_auc_score(y_true=y_cv,
                                                                 y_score=log_regr_clf.predict_proba(X_cv)[:, 1]))
        mlflow.log_metric("cv_test_auc", cv_test_auc)
        mlflow.log_metric("test_auc", skl_metrics.roc_auc_score(y_true=y_test,
                                                                y_score=log_regr_clf.predict_proba(X_test)[:, 1]))

        # Sensitivity/specificity
        test_sensitivity, \
        test_specificity = __calculate_sens_and_spec(y_true=y_test, y_pred=y_test_pred)

        mlflow.log_metric("test_sensitivity", test_sensitivity)
        mlflow.log_metric("test_specificity", test_specificity)

        # Log count of non-zero features
        mlflow.log_metric("count_non_zero_features", np.sum(log_regr_clf.coef_[0] != 0))

        # Define the path for outputting the feature list
        feature_list_output_path = f'/tmp/mlflow_artifacts/{mlflow_run.info.run_id}/feature_list.txt'

        # Create the temporary path to store artifacts
        os.makedirs(os.path.dirname(feature_list_output_path))

        # Write the feature list to file
        with open(feature_list_output_path, 'w') as f:
            for curr_feature in X_cv.columns[(log_regr_clf.coef_[0] != 0)]:
                f.write(curr_feature + "\n")

        # Log the feature list artifact
        mlflow.log_artifact(feature_list_output_path)

        # Confusion matrix
        # Define the path for outputting the feature list
        confusion_output_path = f'/tmp/mlflow_artifacts/{mlflow_run.info.run_id}/confusion_matrix.csv'

        confusion_matrix = skl_metrics.confusion_matrix(y_true=y_test, y_pred=y_test_pred)

        # Output a dataframe with the confusion matrix results
        pd.DataFrame(confusion_matrix,
                     columns=['0_pred','1_pred'],
                     index=['0_actual','1_actual'])\
            .to_csv(confusion_output_path)

        # Log the feature list artifact
        mlflow.log_artifact(confusion_output_path)

def run_experiment_without_mlflow(X_cv, X_test, y_cv, y_test, curr_params_dict):
    # Define the model
    log_regr_clf = LogisticRegression(penalty='l1',
                                      C=curr_params_dict['C'],
                                      class_weight=curr_params_dict['class_weight'],
                                      random_state=110,
                                      solver=curr_params_dict['solver'],
                                      max_iter=curr_params_dict['max_iter'],
                                      verbose=0,
                                      n_jobs=None,
                                      l1_ratio=None)

    # Remove the lowest q quantile of genes based on their CoV
    X_cv, X_test = __remove_lowest_quantile_of_genes_based_on_cov(X_train=X_cv,
                                                                  X_test=X_test,
                                                                  q=curr_params_dict['lower_quantile_removed_CoV'])

    # Run SMOTE on the full X_cv,y_cv data
    if curr_params_dict['use_smote']:
        X_cv, y_cv = __smote_oversample(X_cv, y_cv)

    # Train the model on the full CV data
    log_regr_clf.fit(X_cv, y_cv)

    y_test_pred = log_regr_clf.predict(X_test)

    results_dict = {}

    # Log parameters
    results_dict["params.log10_C"] = np.log10(curr_params_dict['C'])
    results_dict["params.log10_max_iter"] = np.log10(curr_params_dict['max_iter'])
    results_dict["params.solver"] = curr_params_dict['solver']
    results_dict["params.class_weight"] = curr_params_dict['class_weight']
    results_dict["params.lower_quantile_removed_CoV"] = curr_params_dict['lower_quantile_removed_CoV']
    results_dict["params.use_smote"] = curr_params_dict['use_smote']

    # Log test metrics
    results_dict["train_accuracy"] = __calculate_model_accuracy(log_regr_clf, X_cv, y_cv)
    results_dict["test_accuracy"] = __calculate_model_accuracy(log_regr_clf, X_test, y_test)
    results_dict["train_auc"] = skl_metrics.roc_auc_score(y_true=y_cv,
                                                          y_score=log_regr_clf.predict_proba(X_cv)[:, 1])
    results_dict["test_auc"] = skl_metrics.roc_auc_score(y_true=y_test,
                                                         y_score=log_regr_clf.predict_proba(X_test)[:, 1])

    # Sensitivity/specificity
    test_sensitivity, \
    test_specificity = __calculate_sens_and_spec(y_true=y_test, y_pred=y_test_pred)

    results_dict["test_sensitivity"] = test_sensitivity
    results_dict["test_specificity"] = test_specificity

    # Log count of non-zero features
    results_dict["count_non_zero_features"] = np.sum(log_regr_clf.coef_[0] != 0)

    # Write the feature list to file
    results_dict['feature_list'] = X_cv.columns[(log_regr_clf.coef_[0] != 0)].tolist()

    # Confusion matrix
    confusion_matrix = skl_metrics.confusion_matrix(y_true=y_test, y_pred=y_test_pred)

    # Output a dataframe with the confusion matrix results
    results_dict['confusion_matrix'] = pd.DataFrame(confusion_matrix,
                                                    columns=['0_pred','1_pred'],
                                                    index=['0_actual','1_actual'])

    return results_dict