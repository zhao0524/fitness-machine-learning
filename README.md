

# Fitness Machine Learning Project

This project analyzes and models sensor data from barbell exercises using machine learning. It provides scripts for data cleaning, feature engineering, model training, and visualization, all tailored for fitness sensor datasets.

## Getting Started

1. **Clone the repository:**
	 ```sh
	 git clone <repo-url>
	 cd data-science-template-main
	 ```
2. **Set up the environment:**
	 - With conda:
		 ```sh
		 conda env create -f environment.yml
		 conda activate base  # or your custom environment name
		 ```
	 - Or with pip:
		 ```sh
		 pip install -r requirements.txt
		 ```
3. **Install Jupyter and other tools if needed:**
	 ```sh
	 pip install jupyter matplotlib seaborn scikit-learn
	 ```

## Data Preparation

- Place your raw MetaMotion sensor CSV files in `data/raw/MetaMotion/`.
- Run the following scripts to process and clean your data:
	```sh
	python src/data/make_dataset.py
	python src/features/remove_outliers.py
	```
- Processed data will be saved in `data/interim/` and `data/processed/`.

## Model Training & Evaluation

- Train and evaluate models with:
	```sh
	python src/models/train_model.py
	```
- You can add or modify algorithms in `src/models/LearningAlgorithms.py`.

## Visualization

- Generate plots and figures using:
	```sh
	python src/visualization/visualize.py
	```
- Adjust plot settings in `src/visualization/plot_settings.py`.

## Notebooks

- Use the `notebooks/` directory for interactive exploration and prototyping with Jupyter.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.
