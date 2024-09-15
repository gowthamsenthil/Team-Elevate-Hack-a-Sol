# Player Performance Analytics and Prediction Challenge

## Overview

This project focuses on analyzing and predicting the performance of a cricket player based on historical match data from the Indian Premier League (IPL). Using ball-to-ball data from 2008 to 2024, the goal is to build a predictive model that forecasts a player's performance in future matches.

## Team Information

- **Team Name**: Elevate

- **Team Members**:
  - **Chethan Patel**
  - **Gowtham S**

## Dataset

### IPL Ball-to-Ball Information (2008-2024)

[Download Dataset](https://drive.google.com/drive/folders/1_Mb-XrfI-iHZseP-oJ2ngpXmkKKgQ9ur?usp=sharing)


The dataset consists of detailed ball-to-ball data from the IPL spanning from the 2008 season to 2024. This comprehensive dataset includes the following fields:

- **match_id**: Unique identifier for each match.
- **inning**: Indicates whether it's the first or second inning.
- **batting_team**: Name of the team currently batting.
- **bowling_team**: Name of the team currently bowling.
- **over**: The over number within the inning.
- **ball**: The specific ball number within the over.
- **batter**: Name of the batter facing the delivery.
- **bowler**: Name of the bowler delivering the ball.
- **non_striker**: Name of the non-striker batter.
- **batsman_runs**: Runs scored by the batter on that ball.
- **extra_runs**: Extra runs awarded (e.g., wides, no-balls).
- **total_runs**: Total runs scored on the delivery (batsman runs + extra runs).
- **extras_type**: Type of extra run (e.g., wide, no-ball).
- **is_wicket**: Indicates if the delivery resulted in a wicket.
- **player_dismissed**: Name of the player who got out.
- **dismissal_kind**: Type of dismissal (e.g., bowled, caught).
- **fielder**: Name of the fielder involved in the dismissal (if applicable).

### Usage

This dataset provides granular insights into every delivery made during IPL matches, enabling detailed analysis of player performances, match dynamics, and game strategies.

## Tech Usability

This project utilizes the following technologies and tools:

- **Streamlit**: For creating interactive web applications and dashboards.
- **Pandas**: For data manipulation and analysis.
- **JSON**: For data serialization and configuration management.
- **st_aggrid**: For enhanced data grid functionalities within Streamlit apps.
- **Scikit-learn**: For machine learning model implementation and preprocessing:
  - **Train-Test Split**: For dataset splitting.
  - **OneHotEncoder**: For encoding categorical variables.
  - **StandardScaler**: For feature scaling.
  - **MultiOutputRegressor**: For handling multiple target variables.
  - **RandomForestRegressor**: For building predictive models.
  - **Pipeline**: For creating streamlined workflows.
  - **ColumnTransformer**: For applying preprocessing steps.
- **Pickle**: For saving and loading trained models.
- **Plotly**: For creating interactive visualizations.

These tools collectively support the project's data analysis, predictive modeling, and visualization requirements.


## Predictive Modeling

### Model Overview

The project utilizes a MultiOutputRegressor with RandomForestRegressor to predict cricket player performance based on historical match data. This ensemble approach uses multiple decision trees to provide robust and accurate predictions for various metrics such as total runs, boundaries, and strike rate.

### Suitability and Benefits

- **Historical Data Utilization**: The model learns patterns and trends from past matches, providing insights into player performance.
- **Versatility**: Handles complex relationships and interactions within the data, useful for cricket performance prediction.
- **Accuracy and Robustness**: Known for its accuracy and ability to handle noisy data, making it ideal for performance prediction.
- **Scalability**: Can be adapted to include additional features or predict performance for different players and teams.
- **Customizable Output**: Integrates preprocessing steps like feature scaling and encoding, ensuring consistent performance.

### Application

The predictive model forecasts a player's performance in upcoming matches by analyzing trends from previous games. This helps teams and analysts make informed decisions and optimize player performance based on historical insights.

