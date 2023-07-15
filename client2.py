import pickle
import re
import warnings
import flwr as fl
import numpy as np
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, log_loss, roc_auc_score

import utils

def parse_classification_report(report):
    lines = report.split("\n")
    
    # Check if the report has the standard format (precision, recall, f1-score)
    if len(lines) >= 5:
        precision = float(re.findall(r"\d+\.\d+", lines[-4])[0])
        recall = float(re.findall(r"\d+\.\d+", lines[-3])[0])
        f1_score = float(re.findall(r"\d+\.\d+", lines[-2])[0])
    else:
        # If the report has the extended format, extract values from the macro average line
        precision = float(re.findall(r"\d+\.\d+", lines[-2])[0])
        recall = float(re.findall(r"\d+\.\d+", lines[-2])[1])
        f1_score = float(re.findall(r"\d+\.\d+", lines[-2])[2])
    
    return precision, recall, f1_score

# Initialize lists to store values of each round

accuracy_values = []
precision_values = []
recall_values = []
f1_score_values = []
auc_values = []
data_distribution_values = []
communication_overhead_values = []
privacy_metrics_values = []
model_loss_values = []
if __name__ == "__main__":

    (X_train, y_train), (X_test, y_test) = utils.load_data_client1()

    # Split train set into 10 partitions and randomly use one for training.
    partition_id = np.random.choice(10)
    (X_train, y_train) = utils.partition(X_train, y_train, 10)[partition_id]

    # Create LogiticRegression Model
    model = LogisticRegression(
        solver= 'saga',
        penalty="l1",
        max_iter=1,  # local epoch
        # random_state=42,
        warm_start=True,  # prevent refreshing weights when fitting
    )
  
    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model)

    # Define Flower client
    class MnistClient(fl.client.NumPyClient):
        def get_parameters(self,config):  # type: ignore
            return utils.get_model_parameters(model)
        

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['rnd']}")
            return utils.get_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            preds = model.predict(X_test)
            accuracy = (preds == y_test).mean()
            report = classification_report(y_test, preds)
            print(f"Evaluation result: Accuracy: {accuracy}")
            print(f"Classification Report:\n{report}")

            precision, recall, f1_score = parse_classification_report(report)
            

            proba = model.predict_proba(X_test)
            auc = roc_auc_score(y_test, proba[:, 1])
            

            data_distribution = np.bincount(y_train)
            

            communication_overhead = len(X_train) * sys.getsizeof(parameters)
            

            privacy_metrics = utils.calculate_privacy_metrics(X_train, y_train, model)
            
            # Calculate model loss (log loss)
            model_loss = log_loss(y_test, proba)


            # Save accuracy
            accuracy_values.append(accuracy)

            # Extract precision, recall, F1-score from report
            precision_values.append(precision)
            recall_values.append(recall)
            f1_score_values.append(f1_score)

            # Calculate AUC
            auc_values.append(auc)

            # Save data distribution
            data_distribution_values.append(data_distribution)

            # Save communication overhead
            communication_overhead_values.append(communication_overhead)

            # Save privacy metrics
            privacy_metrics_values.append(privacy_metrics)

            # Calculate model loss (log loss)
            model_loss_values.append(model_loss)


            return -1.0 * accuracy, len(X_test), {"accuracy": accuracy}


    # Start Flower client
    fl.client.start_numpy_client(
        #server_address = "localhost:"+ str(sys.argv[1]), 
        server_address = "localhost:5040",
        client=MnistClient())
    # Save the values to pickle files
    with open("C1_accuracy.pickle", "wb") as file:
        pickle.dump(accuracy_values, file)

    with open("C1_precision.pickle", "wb") as file:
        pickle.dump(precision_values, file)

    with open("C1_recall.pickle", "wb") as file:
        pickle.dump(recall_values, file)

    with open("C1_f1_score.pickle", "wb") as file:
        pickle.dump(f1_score_values, file)

    with open("C1_auc.pickle", "wb") as file:
        pickle.dump(auc_values, file)

    with open("C1_data_distribution.pickle", "wb") as file:
        pickle.dump(data_distribution_values, file)

    with open("C1_communication_overhead.pickle", "wb") as file:
        pickle.dump(communication_overhead_values, file)

    with open("C1_privacy_metrics.pickle", "wb") as file:
        pickle.dump(privacy_metrics_values, file)

    with open("C1_model_loss.pickle", "wb") as file:
        pickle.dump(model_loss_values, file)