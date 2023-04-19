# FRIENDS: The One with ML

## Objective
* Analyze and visualize FRIENDS Series
* Classify who says the line among FRIENDS main character
* Build and deploy a web app

## Data
* FRIENDS Scripts (Season 1 to Season 10)

## Tech Stack
* Python, Jupyter Lab, Pandas, Matplotlib, Scikit-learn, Streamlit, Docker

## Tasks
```sh
# Clone the repository
git clone https://github.com/Mregojos/FTOWML-Project
cd FTOWML-Project

# Build and run the web app
docker build -t ftowml-web-app .
docker run --name ftowml-web-app -p 8501:8501 ftowml-web-app
```

```sh
# Run the web app with volume and jupyterlab
# Build and run the web app
docker build -t ftowml-web-app .
docker run --name ftowml-web-app -p 8501:8501 -v $(pwd):/app ftowml-web-app

cd jupyterlab-docker
docker build -t jupyterlab .
cd ..
docker run --name jupyterlab -p 8888:8888 -v $(pwd):/app jupyterlab

# Remove containers
docker rm -f ftowml-web-app
docker rm -f jupyterlab
```

## Reference
