import streamlit as st
import pandas as pd
import json
from st_aggrid import AgGrid
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pickle
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="IPL Player Performance Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

st.markdown("""
<style>

[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetric"] {
    background-color: #262730;
    text-align: center;
    padding: 15px 0;
    text-color: #5d3185;  /* Text color for the metrics */
    border: 2px solid #FF4B4B;  /* Border color for the card */
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

[data-testid="stMetricDeltaIcon-Up"], [data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

</style>
""", unsafe_allow_html=True)


# Add custom CSS to hide the sidebar by default
hide_sidebar_style = """
    <style>
    [data-testid="collapsedControl"] {
        display: none;
    }
    </style>
"""

# Inject custom CSS
st.markdown(hide_sidebar_style, unsafe_allow_html=True)

# Load the datasets
players = pd.read_csv('../dataset/players.csv')
teams = pd.read_csv('../dataset/teams.csv')
data = pd.read_csv('../dataset/deliveries.csv')
batters = pd.read_csv('../dataset/batters.csv')
bowlers = pd.read_csv('../dataset/bowlers.csv')
model_input = pd.read_csv('../predict/streamlit_X_test_batsmen_24.csv')
model_input.drop('Unnamed: 0', axis =1, inplace=True)
bowler_data = pd.read_csv('../predict/streamlit_x_test_bowler_24.csv')
bowler_data = bowler_data.drop('Unnamed: 0', axis=1)
raw_data = pd.read_csv('../dataset/raw_data.csv')

# Specify the path to your .pkl file
file_path = '../models/batsmen_08_23.pkl'

# Load the model
with open(file_path, 'rb') as file:
    model = pickle.load(file)

# Now you can use the model for predictions or other tasks

# Specify the path to your .pkl file
file_path = '../models/bowler_08_23.pkl'

# Load the model
with open(file_path, 'rb') as file:
    model1 = pickle.load(file)


# Load the opposite player-team mapping
with open('../dataset/opposite_player_team_mapping.json') as f:
    opposite_player_team_mapping = json.load(f)

# Load the opposite player-team mapping
with open('../dataset/batsman_teams.json') as f:
    batsman_teams = json.load(f)

# Define the function to calculate batter statistics
def calculate_batter_statistics(df, batter, opponent_team):
    # Filter the DataFrame for the specific batter and opponent team
    filtered_df = df[(df['batter'] == batter) & (df['bowling_team'] == opponent_team)]

    # Calculate total runs scored
    total_runs = filtered_df['batsman_runs'].sum()
    
    # Calculate runs scored through boundaries (4s and 6s)
    boundary_runs = filtered_df[filtered_df['batsman_runs'].isin([4, 6])]['batsman_runs'].sum()
    
    # Calculate total number of balls faced
    balls_faced = filtered_df.shape[0]
    
    # Calculate boundary percentage
    boundary_percentage = (boundary_runs / total_runs * 100) if total_runs > 0 else 0
    
    # Calculate number of centuries
    centuries = (filtered_df.groupby(['match_id', 'inning'])
                  .apply(lambda x: (x['total_runs'].sum() >= 100).sum())
                  .sum())
    
    # Calculate number of half-centuries
    half_centuries = (filtered_df.groupby(['match_id', 'inning'])
                       .apply(lambda x: (50 <= x['total_runs'].sum() < 100).sum())
                       .sum())
    
    # Calculate number of matches played
    matches_played = filtered_df['match_id'].nunique()
    
    return {
        'Total Runs Scored': total_runs,
        'Boundary Percentage': boundary_percentage,
        'Centuries': centuries,
        'Half-Centuries': half_centuries,
        'Matches Played': matches_played,
        'Balls Faced': balls_faced
    }


def calculate_bowler_statistics(df, bowler, opponent_team):
    # Filter the DataFrame for the specific bowler and opponent team
    filtered_df = df[(df['bowler'] == bowler) & (df['batting_team'] == opponent_team)]

    # Calculate total wickets taken
    total_wickets = filtered_df['is_wicket'].sum()

    # Calculate total runs conceded
    total_runs_conceded = filtered_df['total_runs'].sum()

    # Calculate total overs bowled (each unique over)
    total_overs_bowled = filtered_df.groupby(['match_id', 'over']).ngroup().nunique()

    # Calculate average runs per over (Economy Rate)
    economy_rate = (total_runs_conceded / total_overs_bowled) if total_overs_bowled > 0 else 0

    # Calculate total maidens (overs where 0 runs were conceded)
    maiden_overs = filtered_df.groupby(['match_id', 'over']).apply(lambda x: x['total_runs'].sum() == 0).sum()

    # Calculate five-wicket hauls (bowler takes 5 or more wickets in a match)
    five_wicket_hauls = filtered_df.groupby('match_id')['is_wicket'].sum().apply(lambda x: x >= 5).sum()

    return {
        'Total Wickets': total_wickets,
        'Total Runs Conceded': total_runs_conceded,
        'Economy Rate': f"{economy_rate:.2f}",
        'Total Overs Bowled': total_overs_bowled,
        'Maidens': maiden_overs,
        'Five-Wicket Hauls': five_wicket_hauls
    }


def add_multi_index(input_df):
    # Define the multi-level index ('win'/'loss' and 'home'/'away')
    index = pd.MultiIndex.from_product(
        [['Win', 'Loss'], ['Home', 'Away']],
        names=['Result', 'Location']
    )
    
    # Ensure the input DataFrame has enough rows to match the index
    if len(input_df) != len(index):
        raise ValueError(f"Input DataFrame must have {len(index)} rows to match the multi-level index.")

    # Assign the new multi-level index to the DataFrame
    input_df.index = index

    return input_df


# Set up the layout
with st.sidebar:
    st.title("IPL Player Performance Dashboard")

    # player_type = st.selectbox('As a', ['Batsman','Bowler'])

    # if player_type == 'Batsman':
    #     player = st.selectbox("Player", list(batters['batter']))
    # else:
    #     player = st.selectbox("Player", list(bowlers['bowler']))

    # st.markdown("<h3 style='text-align: center;'>VS</h3>", unsafe_allow_html=True)

    # if player_type == 'Bowler':
    #     if player in opposite_player_team_mapping:
    #         available_teams = opposite_player_team_mapping[player]
    #         team = st.selectbox("Opponent Team", available_teams)
    #     else:
    #         st.selectbox("Team", ["No teams available"], index=0)
    #         team = None
    # else:
    #     if player in batsman_teams:
    #         available_teams = batsman_teams[player]
    #         team = st.selectbox("Opponent Team", available_teams)
    #     else:
    #         st.selectbox("Team", ["No teams available"], index=0)
    #         team = None


# Create the layout with columns
col0, col1, col2, col3 = st.columns([1, 3, 1, 3])

with col0:
    player_type = st.selectbox('As a', ['Batsman','Bowler'])

# Add the player dropdown in the first column
with col1:
    if player_type == 'Batsman':
        player = st.selectbox("Player", list(batters['batter']))
    else:
        player = st.selectbox("Player", list(bowlers['bowler']))


# Add the 'VS' text in the middle column
with col2:
    st.markdown("<h3 style='text-align: center;'>VS</h3>", unsafe_allow_html=True)

# Filter the team dropdown based on selected player
with col3:
    if player_type == 'Bowler':
        if player in opposite_player_team_mapping:
            available_teams = opposite_player_team_mapping[player]
            team = st.selectbox("Opponent Team", available_teams)
        else:
            st.selectbox("Team", ["No teams available"], index=0)
            team = None
    else:
        if player in batsman_teams:
            available_teams = batsman_teams[player]
            team = st.selectbox("Opponent Team", available_teams)
        else:
            st.selectbox("Team", ["No teams available"], index=0)
            team = None

st.write("                                              ")

# Define two columns, one for info (metrics) and one for predictions
info1, predictions, info2 = st.columns([1, 3, 1])

# Call the function when a player and team are selected
if player and team:

    if player_type == 'Batsman':

        stats = calculate_batter_statistics(data, player, team)

        with info1:
            # Display the statistics using st.metric in a neat layout
            st.metric("Total Runs Scored", stats['Total Runs Scored'])
            st.metric("Boundary Percentage", f"{stats['Boundary Percentage']:.2f}%")
            st.metric("Centuries", stats['Centuries'])

        with info2:
            st.metric("Half-Centuries", stats['Half-Centuries'])
            st.metric("Matches Played", stats['Matches Played'])
            st.metric("Balls Faced", stats['Balls Faced'])

    else:

        stats = calculate_bowler_statistics(data, player, team)

        with info1:
            # Display the statistics using st.metric in a neat layout for bowlers
            # col1, col2, col3 = st.columns(3)

            st.metric("Total Wickets", stats['Total Wickets'])
            # Ensure the Economy Rate is a float before formatting
            st.metric("Economy Rate", f"{float(stats['Economy Rate']):.2f}")
            st.metric("Total Overs Bowled", stats['Total Overs Bowled'])

        with info2:
            
            st.metric("Total Runs Conceded", stats['Total Runs Conceded'])
            st.metric("Maidens", stats['Maidens'])
            st.metric("Five-Wicket Hauls", stats['Five-Wicket Hauls'])

 
    with predictions:

        st.markdown(f"""
            <div style="text-align: center;">
                <h5>Predictions for {player} against {team}</h5>
            </div>
        """, unsafe_allow_html=True)

        if player_type == 'Batsman':
            # Filter the main DataFrame based on the selected player and opponent strength
            filtered_df = model_input[(model_input['batter'] == player) & (model_input['bowling_team'] == team)]
            
            # Step 1: Select the relevant columns
            selected_columns = ['batter', 'current_form_strike_rate', 'current_form_fours', 'opponent_strength', 
                                'career_avg_strike_rate', 'career_avg_sixes', 'avg_boundary_percentage', 
                                'current_form_sixes', 'career_avg_total_runs', 'is_home_match', 
                                'current_form_total_runs', 'is_win']

            df = filtered_df[selected_columns]

            # Step 2: Group by 'is_home_match' and 'is_win', and average numeric columns
            grouped = df.groupby(['is_home_match', 'is_win']).agg({
                'current_form_strike_rate': 'mean',
                'current_form_fours': 'mean',
                'opponent_strength': 'mean',
                'career_avg_strike_rate': 'mean',
                'career_avg_sixes': 'mean',
                'avg_boundary_percentage': 'mean',
                'current_form_sixes': 'mean',
                'career_avg_total_runs': 'mean',
                'current_form_total_runs': 'mean'
            }).reset_index()

            # Step 3: Create all possible combinations of 'is_home_match' and 'is_win'
            combinations = pd.DataFrame({
                'is_home_match': [True, True, False, False],
                'is_win': [True, False, True, False]
            })

            # Step 4: Merge the grouped DataFrame with the combinations to ensure all (TT, TF, FT, FF) exist
            final_df = pd.merge(combinations, grouped, on=['is_home_match', 'is_win'], how='left')

            # Step 5: Fill missing values with the mean for each numeric column
            final_df = final_df.apply(lambda col: col.fillna(col.mean()) if col.dtype in ['float64', 'int64'] else col)

            # Step 6: Add back the 'batter' column with some representative value (if necessary)
            final_df['batter'] = df['batter'].iloc[0]

            op = pd.DataFrame(model.predict(final_df[['batter',
            'current_form_strike_rate',
            'current_form_fours',
            'opponent_strength',
            'career_avg_strike_rate',
            'career_avg_sixes',
            'avg_boundary_percentage',
            'current_form_sixes',
            'career_avg_total_runs',
            'is_home_match',
            'current_form_total_runs',
            'is_win']]))
    
            # Rename columns for display
            op.columns = ['Total Runs', 'Fours', 'Sixes', 'Strike Rate', 'Boundary Percent', 'Century Probability', 'Half Century Probability']

            # Round 'Total Runs', 'Fours', and 'Sixes' to nearest integer
            op[['Total Runs', 'Fours', 'Sixes']] = op[['Total Runs', 'Fours', 'Sixes']].round().astype(int)

            # Format the DataFrame
            op['Strike Rate'] = op['Strike Rate'].round(2)
            op['Boundary Percent'] = op['Boundary Percent'].round(2)
            op['Century Probability'] = (op['Century Probability'] * 100).round(2)
            op['Half Century Probability'] = (op['Half Century Probability'] * 100).round(2)

            op = add_multi_index(op)

            raw_data = raw_data[(raw_data['batter'] == player) & (raw_data['bowling_team'] == team)]

            # Assuming raw_data is your DataFrame
            raw_data['date'] = pd.to_datetime(raw_data['date'])  # Convert 'date' to datetime
            raw_data['year'] = raw_data['date'].dt.year  # Extract year

            # Aggregate total runs scored by each batter per year
            agg_df = raw_data.groupby(['year', 'batter'])['batsman_runs'].sum().reset_index()

            # Create the line plot
            fig = px.line(agg_df, x='year', y='batsman_runs', color='batter',
                        labels={'year': 'Year', 'batsman_runs': 'Total Runs'},
                        markers=True)

            

        
           # Format the DataFrame to show 2 decimal places and add % symbol for Century
            op = op.style.format({
                'Strike Rate': '{:.2f}',
                'Boundary Percent': '{:.2f}%',
                'Century Probability': '{:.2f}%',
                'Half Century Probability': '{:.2f}%'
            })

            st.table(op)

            test1, test2 = st.columns([1,1])

            with test1:
                st.plotly_chart(fig)

            
        if player_type == 'Bowler': 

            # Filter the main DataFrame based on the selected player and opponent strength
            filtered_df = bowler_data[(bowler_data['bowler'] == player) & (bowler_data['batting_team'] == team)]

            # Step 1: Select the relevant columns for bowlers
            selected_columns = ['bowler', 'current_form_wickets', 'current_form_economy_rate', 'current_form_maidens', 
                                'current_form_runs_conceded', 'current_form_balls_bowled', 'career_avg_wickets', 
                                'career_avg_economy_rate', 'career_avg_maidens', 'career_avg_runs_conceded', 
                                'career_avg_balls_bowled', 'opponent_strength', 'maiden_percentage', 
                                'avg_maiden_percentage', 'is_win', 'is_home_match']

            df = filtered_df[selected_columns]

            # Filter the main DataFrame based on the selected player and opponent strength
            
            # Step 2: Group by 'is_home_match' and 'is_win', and average numeric columns
            # You can use agg to average numeric columns, skipping the 'bowler' column.
            grouped = df.groupby(['is_home_match', 'is_win']).agg({
                'current_form_wickets': 'mean',
                'current_form_economy_rate': 'mean',
                'current_form_maidens': 'mean',
                'current_form_runs_conceded': 'mean',
                'current_form_balls_bowled': 'mean',
                'career_avg_wickets': 'mean',
                'career_avg_economy_rate': 'mean',
                'career_avg_maidens': 'mean',
                'career_avg_runs_conceded': 'mean',
                'career_avg_balls_bowled': 'mean',
                'opponent_strength': 'mean',
                'maiden_percentage': 'mean',
                'avg_maiden_percentage': 'mean'
            }).reset_index()

            # Step 3: Create all possible combinations of 'is_home_match' and 'is_win'
            combinations = pd.DataFrame({
                'is_home_match': [True, True, False, False],
                'is_win': [True, False, True, False]
            })

            # Step 4: Merge the grouped DataFrame with the combinations to ensure all (TT, TF, FT, FF) exist
            final_df = pd.merge(combinations, grouped, on=['is_home_match', 'is_win'], how='left')

            # Step 5: Fill missing values with the mean for each numeric column
            final_df = final_df.apply(lambda col: col.fillna(col.mean()) if col.dtype in ['float64', 'int64'] else col)

            # Step 6: Add back the 'bowler' column with some representative value (if necessary)
            # You might want to insert the bowler name back, e.g., first bowler name
            final_df['bowler'] = df['bowler'].iloc[0]

            bowlers_predict = pd.DataFrame(model1.predict(final_df[['bowler', 'current_form_wickets', 'current_form_economy_rate', 'current_form_maidens', 
                    'current_form_runs_conceded', 'current_form_balls_bowled', 'career_avg_wickets', 
                    'career_avg_economy_rate', 'career_avg_maidens', 'career_avg_runs_conceded', 
                    'career_avg_balls_bowled', 'opponent_strength', 'maiden_percentage', 
                    'avg_maiden_percentage', 'is_win', 'is_home_match']]))

            bowlers_predict.columns = ['Wickets', 'Runs_conceded', 'Balls Bowled', 'Economy Rate', 'Maidens','Extras']

            # Round 'Total Runs', 'Fours', and 'Sixes' to nearest integer
            bowlers_predict[['Wickets', 'Runs_conceded', 'Balls Bowled', 'Maidens','Extras']] = bowlers_predict[['Wickets', 'Runs_conceded', 'Balls Bowled', 'Maidens','Extras']].round().astype(int)

            # Format the DataFrame
            bowlers_predict['Economy Rate'] = bowlers_predict['Economy Rate'].round(2)

            bowlers_predict = add_multi_index(bowlers_predict)

            st.table(bowlers_predict)

            # Format the DataFrame to show 2 decimal places
            bowlers_predict = bowlers_predict.style.format({
                'Economy Rate': '{:.2f}',
            })


 



                    