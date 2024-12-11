# %%
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

file_path = "cleaned_kaggle_data.csv"
data = pd.read_csv(file_path)

def clean_salary(salary):
    if isinstance(salary, str):
        salary = salary.replace("$", "").replace(",", "").strip()
        if "-" in salary:
            lower, upper = salary.split("-")
            return (int(lower) + int(upper)) / 2
        elif salary.isdigit():
            return int(salary)
    return np.nan

data["Salary"] = data["Salary"].apply(clean_salary)

if data["Salary"].isna().sum() > 0:
    print(f"Rows with NaN Salary: {data['Salary'].isna().sum()}")

data = data.dropna(subset=["Salary"])

original_country_column = data["in_which_country_do_you_currently_reside"].copy()

job_titles = [
    "Data Scientist",
    "Software Engineer",
    "Research Scientist",
    "Developer Advocate",
    "Data Analyst (Business, Marketing, Financial, Quantitative, etc)",
    "Data Engineer",
    "Machine Learning/ MLops Engineer",
    "Engineer (non-software)",
    "Teacher / professor",
    "Other"
]

language_columns = [
    "what_programming_languages_do_you_use_on_a_regular_basis_select_all_that_apply__selected_choice__python",
    "what_programming_languages_do_you_use_on_a_regular_basis_select_all_that_apply__selected_choice__r",
    "what_programming_languages_do_you_use_on_a_regular_basis_select_all_that_apply__selected_choice__sql",
    "what_programming_languages_do_you_use_on_a_regular_basis_select_all_that_apply__selected_choice__c",
    "what_programming_languages_do_you_use_on_a_regular_basis_select_all_that_apply__selected_choice__java",
    "what_programming_languages_do_you_use_on_a_regular_basis_select_all_that_apply__selected_choice__javascript",
    "what_programming_languages_do_you_use_on_a_regular_basis_select_all_that_apply__selected_choice__bash",
    "what_programming_languages_do_you_use_on_a_regular_basis_select_all_that_apply__selected_choice__php",
    "what_programming_languages_do_you_use_on_a_regular_basis_select_all_that_apply__selected_choice__matlab",
    "what_programming_languages_do_you_use_on_a_regular_basis_select_all_that_apply__selected_choice__julia",
    "what_programming_languages_do_you_use_on_a_regular_basis_select_all_that_apply__selected_choice__go",
    "what_programming_languages_do_you_use_on_a_regular_basis_select_all_that_apply__selected_choice__none",
    "what_programming_languages_do_you_use_on_a_regular_basis_select_all_that_apply__selected_choice__other"
]

existing_language_columns = [col for col in language_columns if col in data.columns]

data = pd.get_dummies(
    data,
    columns=[
        "what_is_your_gender__selected_choice",
        "in_which_country_do_you_currently_reside",
        "what_is_your_age__years",
        "select_the_title_most_similar_to_your_current_role_or_most_recent_title_if_retired__selected_choice"
    ],
    drop_first=True
)

X = data[
    existing_language_columns
    + [col for col in data.columns if col.startswith("what_is_your_gender__selected_choice_")]
    + [col for col in data.columns if col.startswith("in_which_country_do_you_currently_reside_")]
    + [col for col in data.columns if col.startswith("what_is_your_age__years_")]
    + [col for col in data.columns if col.startswith("select_the_title_most_similar_to_your_current_role_or_most_recent_title_if_retired__selected_choice_")]
]
y = data["Salary"]

if y.isna().sum() > 0:
    print(f"Remaining NaN values in y: {y.isna().sum()}")
    raise ValueError("Target variable 'Salary' contains NaN values.")

X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

st.title("Data Science Salary Predictor")

primary_languages = st.multiselect(
    "What programming languages do you use on a regular basis?",
    [
        "Python", "R", "SQL", "C", "Java", "JavaScript", "Bash", "PHP", "MATLAB", "Julia", "Go", "None", "Other"
    ]
)
gender = st.selectbox(
    "What is your gender?",
    ["Male", "Female", "Non-binary", "Prefer not to say"]
)
country = st.selectbox(
    "In which country do you currently reside?",
    original_country_column.unique()
)
age = st.selectbox(
    "What is your age group?",
    data.filter(like="what_is_your_age__years_").columns.str.replace(
        "what_is_your_age__years_", ""
    )
)
job_title = st.selectbox(
    "What is your current or most recent job title?",
    job_titles
)

input_data = pd.DataFrame({
    **{col: [1] if col.split("__")[-1].lower() in [lang.lower() for lang in primary_languages] else [0] for col in existing_language_columns},
    **{col: [1] if f"what_is_your_gender__selected_choice_{gender}" == col else [0] for col in X.columns if col.startswith("what_is_your_gender__selected_choice_")},
    **{col: [1] if f"in_which_country_do_you_currently_reside_{country}" == col else [0] for col in X.columns if col.startswith("in_which_country_do_you_currently_reside_")},
    **{col: [1] if f"what_is_your_age__years_{age}" == col else [0] for col in X.columns if col.startswith("what_is_your_age__years_")},
    **{col: [1] if f"select_the_title_most_similar_to_your_current_role_or_most_recent_title_if_retired__selected_choice_{job_title}" == col else [0] for col in X.columns if col.startswith("select_the_title_most_similar_to_your_current_role_or_most_recent_title_if_retired__selected_choice_")}
})

input_data = input_data.reindex(columns=X.columns, fill_value=0)

if st.button("Predict"):
    predicted_salary = model.predict(input_data)[0]
    st.subheader(f"Predicted Salary: ${predicted_salary:,.2f}")

# %%
