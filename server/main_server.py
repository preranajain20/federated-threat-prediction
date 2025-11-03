import logging
import flwr as fl
from secure_aggregation import SecureAggregationStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_server():
    strategy = SecureAggregationStrategy(
        secure_mode="simulated_secure",
        robust_agg="trimmed_mean",
        trimmed_fraction=0.1,
        krum_f=1,
        min_fit_clients=2,
        min_available_clients=2
    )
    fl.server.start_server("0.0.0.0:8080", config={"num_rounds": 5}, strategy=strategy)

if __name__ == "__main__":
    logger.info("Starting FL Server...")
    run_server()
