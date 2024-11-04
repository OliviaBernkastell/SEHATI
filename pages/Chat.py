import os
import pandas as pd
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationEntityMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from serpapi import GoogleSearch

# Replace with your actual paths or remove if not using an image
CHATBOT_PROFILE_PICTURE = "chatbot_profile.png"

# Replace with your actual API keys
os.environ["GOOGLE_API_KEY"] = "AIzaSyCSTaIxbxuHj05MK0fvFO30cRUgN05z2h4"
SERPAPI_API_KEY = "12a454ecbc2ec1b425a9f71fa06272861f62f271f430e600213c98ae88b1cb01"

# Define the `get_answer_box` tool function
def get_answer_box(query):
    search = GoogleSearch({
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "hl": "id",
    })
    result = search.get_dict()
    if 'answer_box' in result:
        return result['answer_box']
    elif 'organic_results' in result and len(result['organic_results']) > 0:
        first_result = result['organic_results'][0]
        title = first_result.get("title", "No title available")
        snippet = first_result.get("snippet", "No snippet available")
        link = first_result.get("link", "No link available")
        return f"Title: {title}\nSnippet: {snippet}\nLink: {link}"
    else:
        return "No relevant information found."

# Create the Tool for the agent to use
answer_box_tool = Tool(
    name="get_answer_box",
    func=get_answer_box,
    description="Fetches real-time data from a search query, e.g., 'What is the temperature in London today?'"
)

# Initialize the language model for the agent
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Initialize the agent with the tool
tools = [answer_box_tool]
realtime_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Calorie calculation function
def calculate_calories(current_weight, height, age, gender, target_weight, months, activity_level):
    if gender.lower() == 'male':
        BMR = 10 * current_weight + 6.25 * height - 5 * age + 5
    elif gender.lower() == 'female':
        BMR = 10 * current_weight + 6.25 * height - 5 * age - 161
    else:
        raise ValueError("Gender must be 'male' or 'female'")

    activity_multipliers = {
        'sedentary': 1.2,
        'lightly_active': 1.375,
        'moderately_active': 1.55,
        'very_active': 1.725,
        'extra_active': 1.9
    }

    if activity_level.lower() not in activity_multipliers:
        raise ValueError("Invalid activity level.")

    TDEE = BMR * activity_multipliers[activity_level.lower()]
    weight_change = target_weight - current_weight
    total_caloric_change_needed = abs(weight_change) * 7700
    days = months * 30
    daily_caloric_surplus_or_deficit = total_caloric_change_needed / days

    if weight_change > 0:
        daily_caloric_intake = TDEE + daily_caloric_surplus_or_deficit
    else:
        daily_caloric_intake = TDEE - daily_caloric_surplus_or_deficit

    return daily_caloric_intake, daily_caloric_surplus_or_deficit

# Load the food data
@st.cache_data
def load_food_data():
    food_data_path = 'food_raw.csv'
    food_df = pd.read_csv(food_data_path)
    food_df_selected = food_df[['Nama Bahan Makanan', 'Energi (Kal)', 'Protein (g)', 'Lemak (g)', 'Karbohidrat (g)']]
    food_df_selected.columns = ['name', 'kcal', 'protein', 'fat', 'carbs']  # Renaming columns for easier use
    # Convert the data to a list of dictionaries to match the previous structure
    food_database = food_df_selected.to_dict(orient='records')
    return food_database

# Function to get food recommendations based on targets
def get_food_recommendations(target_kcal, target_protein, target_fat, target_carbs, food_database):
    recommendations = []
    total_kcal = 0
    total_protein = 0
    total_fat = 0
    total_carbs = 0

    for food in food_database:
        # Check if adding this food will stay within target limits
        if (total_kcal + food["kcal"] <= target_kcal and
            total_protein + food["protein"] <= target_protein and
            total_fat + food["fat"] <= target_fat and
            total_carbs + food["carbs"] <= target_carbs):
            recommendations.append(food)
            total_kcal += food["kcal"]
            total_protein += food["protein"]
            total_fat += food["fat"]
            total_carbs += food["carbs"]

    return recommendations, total_kcal, total_protein, total_fat, total_carbs

# Initialize session state
def initialize_session_state():
    if 'api_key' not in st.session_state:
        st.session_state.api_key = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'entity_memory' not in st.session_state:
        st.session_state.entity_memory = ConversationEntityMemory(llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash"))
    if 'awaiting_food_recommendation' not in st.session_state:
        st.session_state.awaiting_food_recommendation = False

# Streamlit app
def chatbot_page():
    st.title("SEHATI: Solusi Energi dan Hidup Sehat Terkini")

    initialize_session_state()

    api_key = st.sidebar.text_input("Enter your API key:", value=st.session_state.api_key, type="password")
    if not api_key:
        st.sidebar.error("Please enter your API key.")
        st.stop()
    else:
        st.session_state.api_key = api_key
    genai.configure(api_key=api_key)

    # Calorie calculation section
    st.header("Calorie Calculation")
    current_weight = st.number_input("Current weight (kg):", min_value=0.0, step=0.1)
    height = st.number_input("Height (cm):", min_value=0.0, step=0.1)
    age = st.number_input("Age (years):", min_value=0, step=1)
    gender = st.selectbox("Gender:", ["male", "female"])
    target_weight = st.number_input("Target weight (kg):", min_value=0.0, step=0.1)
    months = st.number_input("Timeframe to reach target (months):", min_value=1, step=1)
    activity_level = st.selectbox("Activity level:", ["sedentary", "lightly_active", "moderately_active", "very_active", "extra_active"])

    if st.button("Calculate Calories"):
        try:
            caloric_intake, daily_change = calculate_calories(
                current_weight, height, age, gender, target_weight, months, activity_level
            )
            result_message = f"Daily caloric intake: {caloric_intake:.2f} kcal, Daily change: {daily_change:.2f} kcal."
            st.success(result_message)

            # Macronutrient distribution calculation based on caloric_intake only
            protein_calories = caloric_intake * 0.20
            fat_calories = caloric_intake * 0.30
            carbs_calories = caloric_intake * 0.50

            protein_grams = protein_calories / 4  # 4 kcal per gram of protein
            fat_grams = fat_calories / 9         # 9 kcal per gram of fat
            carbs_grams = carbs_calories / 4     # 4 kcal per gram of carbs

            # Format parameters, results, and macronutrient distribution as a single string
            context_message = (
                f"Calorie Calculation Parameters:\n"
                f"Current Weight: {current_weight} kg\n"
                f"Height: {height} cm\n"
                f"Age: {age} years\n"
                f"Gender: {gender}\n"
                f"Target Weight: {target_weight} kg\n"
                f"Timeframe: {months} months\n"
                f"Activity Level: {activity_level}\n\n"
                f"Results:\n"
                f"Caloric Intake: {caloric_intake:.2f} kcal\n"
                f"Daily Change: {daily_change:.2f} kcal\n\n"
                f"Macronutrient Distribution:\n"
                f"Protein: {protein_grams:.2f} g ({protein_calories:.2f} kcal)\n"
                f"Fat: {fat_grams:.2f} g ({fat_calories:.2f} kcal)\n"
                f"Carbs: {carbs_grams:.2f} g ({carbs_calories:.2f} kcal)"
            )

            # Store the formatted message in memory
            st.session_state.entity_memory.save_context(
                {"input": "Calorie Calculation Parameters"},
                {"output": context_message}
            )
            st.session_state.chat_history.append(("You", "Calorie Calculation Result"))
            st.session_state.chat_history.append(("Assistant", result_message))

            # Display the distribution in boxes
            st.header("Macronutrient Distribution")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Protein (20%)", f"{protein_grams:.2f} g")
            with col2:
                st.metric("Fat (30%)", f"{fat_grams:.2f} g")
            with col3:
                st.metric("Carbs (50%)", f"{carbs_grams:.2f} g")

            # Store the calculated values in session state for later use
            st.session_state.caloric_intake = caloric_intake
            st.session_state.protein_grams = protein_grams
            st.session_state.fat_grams = fat_grams
            st.session_state.carbs_grams = carbs_grams
            st.session_state.food_database = load_food_data()

            # Add assistant message to chat history
            assistant_message = "Would you like to receive food recommendations based on your targets?"
            st.session_state.chat_history.append(("Assistant", assistant_message))
            st.session_state.awaiting_food_recommendation = True

        except ValueError as e:
            st.error(f"Error: {e}")

    # Chatbot interaction
    st.header("Chat with SEHATI")

    if st.session_state.chat_history:
        for i, (sender, message) in enumerate(st.session_state.chat_history):
            if sender == "Assistant":
                col1, col2 = st.columns([1, 9])
                with col1:
                    if os.path.exists(CHATBOT_PROFILE_PICTURE):
                        st.image(CHATBOT_PROFILE_PICTURE, width=50)
                with col2:
                    st.write(f"**SEHATI:** {message}")
            else:
                st.write(f"**You:** {message}")

    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Enter your message:", "", key="user_input_key")
        submit_button = st.form_submit_button("Send")

    custom_prompt = PromptTemplate.from_template(
        "The following is a conversation between the assistant and the user.\n\n"
        "Entities to remember: {entities}\n\n"
        "Conversation history:\n{history}\n\n"
        "User: {input}\n\n"
    )

    if submit_button and user_input:
        if st.session_state.get('awaiting_food_recommendation', False):
            # Process the user's response to the food recommendation offer
            if "yes" in user_input.lower():
                # Generate food recommendations
                recommended_foods, total_kcal_foods, total_protein_foods, total_fat_foods, total_carbs_foods = get_food_recommendations(
                    st.session_state.caloric_intake,
                    st.session_state.protein_grams,
                    st.session_state.fat_grams,
                    st.session_state.carbs_grams,
                    st.session_state.food_database
                )
                # Format the recommendations
                recommendations_text = "Here are some food recommendations based on your targets:\n"
                for food in recommended_foods:
                    recommendations_text += f"- {food['name']} (Kcal: {food['kcal']}, Protein: {food['protein']}g, Fat: {food['fat']}g, Carbs: {food['carbs']}g)\n"
                # Add the assistant's response to the chat history
                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("Assistant", recommendations_text))
                st.session_state.awaiting_food_recommendation = False
            else:
                # User does not want recommendations
                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("Assistant", "Alright, let me know if you need anything else."))
                st.session_state.awaiting_food_recommendation = False
        else:
            # Normal chat processing
            st.session_state.chat_history.append(("You", user_input))
            if "google" in user_input.lower() or "internet" in user_input.lower():
                response = realtime_agent.run(user_input)
            else:
                conversation = ConversationChain(
                    llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash"),
                    memory=st.session_state.entity_memory,
                    prompt=custom_prompt,
                    verbose=True
                )
                response = conversation.predict(input=user_input)

            st.session_state.chat_history.append(("Assistant", response))
        st.experimental_rerun()

# Run the Streamlit app
if __name__ == "__main__":
    chatbot_page()
