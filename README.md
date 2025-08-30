## Code Running Instructions:

### Docker

THE SCRIPT...described

# Step 1: Optionally build the base image

# If "--skip-base" is passed as the first argument, skip this step

if [ "$1" != "--skip-base" ]; then
echo "Building base image..."
docker build -f Dockerfile.base -t rl-base:py312-cu118-v1 .
else
echo "Skipping base image build"
fi

# Step 2: Build the hyperparameter tuning image

echo "Building hyperparameter image..."
docker build -f Dockerfile.hyperparameters -t hyperparameter-image .

# Step 3: Create a shared Docker volume to persist data between containers

echo "Creating shared volume..."
docker volume create mydata

# Step 4: Run the hyperparameter container

# - Mounts the host-side configs directory into the container

# - Mounts the shared volume at /app/data

# - Exposes port 8080

# - Loads environment variables from .env

# - Uses all available GPUs

# - Removes the container after it exits

echo "Running hyperparameter container..."
docker run \
 -v "$(pwd)/configs:/app/data/configs" \
 -v mydata:/app/data \
 -p 8080:8080 \
 --env-file .env \
 --gpus all \
 --rm -it hyperparameter-image

# Step 5: Build the reward testing image

echo "Building reward tests image..."
docker build -f Dockerfile.rewardTests -t rewardtests-image .

# Step 6: Run the reward container

# - Uses the same shared volume to access updated configs

# - Exposes port 8090

# - Loads environment variables from .env

# - Uses all available GPUs

# - Removes the container after it exits

echo "Hyperparameter container finished. Starting reward tests..."
docker run \
 -v mydata:/app/data \
 -p 8090:8090 \
 --env-file .env \
 --gpus all \
 --rm -it rewardtests-image

# Final message

echo "All containers completed successfully."

### AWS (if you have money) - you'll likely want to avoid this since GPU-heavy projects are expensive.

`aws configure`
`aws ecr get-login-password --region [region] | docker login --username AWS --password-stdin [account-no].dkr.ecr.[region].amazonaws.com`
`docker build -f Dockerfile.hyperparameters -t rlproject/repositoryname_hypers .`
`docker build -f Dockerfile.rewardTests -t rlproject/repositoryname_rewards .`

`docker tag rlproject/repositoryname_hypers:latest [account-no].dkr.ecr.[region].amazonaws.com/rlproject/repositoryname:hyperparameters`
`docker tag rlproject/repositoryname_rewards:latest [account-no].dkr.ecr.[region].amazonaws.com/rlproject/repositoryname:rewards`
`docker push [account-no].dkr.ecr.[region].amazonaws.com/rlproject/repositoryname:hyperparameters`
`docker push [account-no].dkr.ecr.[region].amazonaws.com/rlproject/repositoryname:rewards`
