from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration
)
from azure.identity import DefaultAzureCredential
import json
import time

with open('config.json', 'r') as f:
    config = json.load(f)

subscription_id=config["subscription_id"]
resource_group=config["resource_group"]
workspace_name=config["workspace_name"]

# get a handle to the workspace
ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace_name
)

# method 1: define an endpoint name
endpoint_name = "nutri-scan-model-1"

# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name = endpoint_name, 
    description="NutriScan Model Endpoint",
    auth_mode="key"
)

# Create endpoint and wait for it to complete
try:
    print("Creating/updating endpoint...")
    endpoint_operation = ml_client.online_endpoints.begin_create_or_update(endpoint)
    endpoint = endpoint_operation.result()  # Use result() to ensure completion and get the result
    print("Endpoint created successfully!")
except Exception as e:
    print(f"Endpoint creation failed: {e}")
    # Try to get any existing endpoint to continue
    try:
        endpoint = ml_client.online_endpoints.get(name=endpoint_name)
    except:
        print("Could not retrieve existing endpoint. Exiting.")
        raise

# Now create the deployment
model = Model(path="models/model_2.pth")
env = Environment(
    conda_file="environment.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
)

deployment = ManagedOnlineDeployment(
    name="nutriscan-1", 
    endpoint_name=endpoint_name,
    model=model,
    environment=env,
    code_configuration=CodeConfiguration(
        code="./scoring",
        scoring_script="score.py"
    ),
    instance_type="Standard_DS2_v2",
    instance_count=1,
)

# Create deployment with more detailed error handling
try:
    print("Creating/updating deployment...")
    deployment_operation = ml_client.online_deployments.begin_create_or_update(deployment)
    # Wait for a bit to get initial deployment status
    time.sleep(30)
    
    # Try to get detailed status
    try:
        status = ml_client.online_deployments.get(name="nutriscan", endpoint_name=endpoint_name)
        print(f"Deployment status: {status.provisioning_state}")
    except Exception as status_error:
        print(f"Could not get deployment status: {status_error}")
    
    # Try to get logs even if deployment is failing
    try:
        print("Attempting to get logs (even during failure)...")
        logs = ml_client.online_deployments.get_logs(
            name="nutriscan", endpoint_name=endpoint_name, lines=100
        )
        print("=== DEPLOYMENT LOGS ===")
        print(logs)
        print("======================")
    except Exception as log_error:
        print(f"Could not retrieve logs: {log_error}")
    
    # Continue with deployment completion
    print("Waiting for deployment to complete...")
    deployment = deployment_operation.result()
    print("Deployment succeeded!")
    
except Exception as e:
    print(f"Deployment failed: {e}")

# Try to get the endpoint details if everything worked
try:
    endpoint = ml_client.online_endpoints.get(name=endpoint_name)
    scoring_uri = endpoint.scoring_uri
    print(f"Endpoint URI: {scoring_uri}")
except Exception as e:
    print(f"Could not get endpoint details: {e}")