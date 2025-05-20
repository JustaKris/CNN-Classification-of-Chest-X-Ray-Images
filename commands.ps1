# Environment setup - Anaconda
conda env list
conda create -p venv python=3.11 -y
conda activate venv/

# Environment setup - Python 3.11
python3.11 -m venv venv
venv\Scripts\activate

# Run app locally
pip install -r requirements.txt
python app.py  # localhost:5000

# Tensorboard
tensorboard --logdir=./logs/tensorboard

# Run unit tests
python -m unittest discover -s tests/unit -p '*_test.py'

# Run ui tests
npx playwright install
npx playwright test

# Docker
docker build -t justakris/chest-xray-classification-app:latest .
docker run -p 5050:5050 -d --name chest-xray-classification justakris/chest-xray-classification-app:latest
docker stop chest-xray-classification

# Docker Hub
docker login
docker tag chest-xray-classification-app justakris/chest-xray-classification-app:latest
docker push justakris/chest-xray-classification-app:latest

# Interacting with container
docker exec -it chest-xray-classification bash

# GitHub Actions and deployment to Render 
# Render Key -> rnd_FTBxo7rYaRqdbL9frCXpSyFuBfsg
# Render ID (from hook in settings) -> srv-cpmjffg8fa8c73ajaajg
# Render ID Docker version -> srv-cpooujmehbks73enrqlg