import csv
import flwr as fl
import utils
import sys
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any
import pandas as pd
import numpy as np
from flwr.common import NDArrays, Scalar
from typing import Dict, Optional, Tuple


def fit_round(rnd: int) -> Dict:
    """Send round number to client."""
    return {"rnd": rnd}


def get_eval_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    _, (X_test, y_test) = utils.load_data()

    # The `evaluate` function will be called after every round
    def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]):
        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        # Use model.predict for binary predictions
        preds = model.predict(X_test)
        loss = log_loss(y_test, preds, labels=[1, 0])
        accuracy = model.score(X_test, y_test)
        res = pd.DataFrame(preds)
        res.index = pd.DataFrame(X_test).index  # it's important for comparison
        res.columns = ["prediction"]
        res["real"] = y_test  # Add the actual labels to the DataFrame
        res.to_csv("prediction_results.csv")

        # Save aggregated results of loss and accuracy in a CSV file
        if server_round == 1:
            mode = "w"  # Create a new file if it doesn't exist
        else:
            mode = "a"  # Append to the existing file

        with open("aggregated_results.csv", mode=mode) as file:
            writer = csv.writer(file)
            if server_round == 1:
                writer.writerow(["Round", "Loss", "Accuracy"])  # Write the header if it's a new file
            writer.writerow([server_round, loss, accuracy])
        return {"Aggregated Results: loss ": loss}, {"accuracy": accuracy}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LogisticRegression(
        solver='saga',
        penalty='l2',
        C=1.0,
        max_iter=1,
        random_state=42,
        warm_start=True,
    )

    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        on_fit_config_fn=fit_round,
        evaluate_fn=get_eval_fn(model),
    )
    fl.server.start_server(
        # server_address="localhost:"+ str(sys.argv[1]),
        server_address="localhost:5040",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=20),  # Increase to a higher value


    )
