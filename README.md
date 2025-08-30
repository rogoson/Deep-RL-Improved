## Code Running Instructions:

### Docker

`docker build -f Dockerfile.base -t rl-base:py312-cu118-v1 .`
`docker build -f Dockerfile.hyperparameters -t hyperparamter-image .`
`docker build -t hyperparameter-image .`
`docker build -f Dockerfile.rewardTests -t rewardtests-image .`
`docker build -t rewardtests-image .`
`docker run -p 8080:8080 --env-file .env --gpus all --rm -it hyperparameter-image` // either one

### AWS (if you have money) - you'll likely want to avoid this since GPU-heavy projects are expensive.

`aws configure`
`aws ecr get-login-password --region [region] | docker login --username AWS --password-stdin [account-no].dkr.ecr.[region].amazonaws.com`
`docker build -f Dockerfile.hyperparameters -t rlproject/repositoryname_hypers .`
`docker build -f Dockerfile.rewardTests -t rlproject/repositoryname_rewards .`

`docker tag rlproject/repositoryname_hypers:latest [account-no].dkr.ecr.[region].amazonaws.com/rlproject/repositoryname:hyperparameters`
`docker tag rlproject/repositoryname_rewards:latest [account-no].dkr.ecr.[region].amazonaws.com/rlproject/repositoryname:rewards`
`docker push [account-no].dkr.ecr.[region].amazonaws.com/rlproject/repositoryname:hyperparameters`
`docker push [account-no].dkr.ecr.[region].amazonaws.com/rlproject/repositoryname:rewards`
