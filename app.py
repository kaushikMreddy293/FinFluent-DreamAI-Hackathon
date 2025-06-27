import streamlit as st
from finance_advisor import agent_executor, get_net_worth, get_financial_goals
import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dotenv import load_dotenv
from data_manager import DataManager
from typing import Dict, List, Any, Optional, Tuple
import rag_service

# Load environment variables
load_dotenv()

# Set page config with theme
st.set_page_config(
    page_title="FinFluent",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set theme configuration
st.markdown("""
    <style>
        .stApp {
            --primary-color: #FF6B35;
            --primary-text-color: #333333;
            --secondary-background-color: #FFF9F5;
            --text-color: #333333;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "👋 Welcome to FinFluent! I'm your personal finance assistant. How can I help you today?"}
    ]
if 'current_page' not in st.session_state:
    st.session_state.current_page = "dashboard"
if 'show_profile_form' not in st.session_state:
    st.session_state.show_profile_form = False
if 'show_transaction_form' not in st.session_state:
    st.session_state.show_transaction_form = False

st.markdown("""
    <style>
/* Main layout background */
.stApp {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
    background-color: #111826;
    color: #ffffff;
}

/* Sidebar */
.sidebar .sidebar-content {
    background-color: #111826;
    color: #ffffff;
}

.main {
    background-color: #111826;
    color: #ffffff;
}

/* Text Input */
.stTextInput>div>div>input {
    border-radius: 20px;
    padding: 12px 20px;
    border: 2px solid #29A63C;
    background-color: #111826;
    color: #ffffff;
}

/* Buttons */
.stButton>button, [data-testid="stButton"]>button {
    width: 100% !important;
    border-radius: 20px !important;
    background: #29A63C !important;
    color: #ffffff !important;
    font-weight: 500 !important;
    border: none !important;
    padding: 8px 16px !important;
    margin: 4px 0 !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: none !important;
    background-image: none !important;
}

.stButton>button:hover {
    background: #208C31 !important;
    box-shadow: 0 4px 12px rgba(41, 166, 60, 0.3) !important;
    transform: translateY(-1px) !important;
}

.stButton>button[kind="secondary"] {
    background: #111826;
    color: #29A63C;
    border: 2px solid #29A63C;
}

.stButton>button[kind="secondary"]:hover {
    background: #1a1f2d;
}

/* Number input +/- buttons */
button[data-baseweb="button"][aria-label="Decrease"],
button[data-baseweb="button"][aria-label="Increase"] {
    width: 30px !important;
    height: 30px !important;
    padding: 0 !important;
    margin: 0 4px !important;
    border-radius: 4px !important;
    background: #2f3540 !important;
    color: #ffffff !important;
}

/* Slider - visual track, knob, label */
.stSlider > div {
    color: #29A63C !important;
}

.stSlider [role="slider"] {
    background-color: #29A63C !important;
    border: 2px solid #29A63C !important;
    box-shadow: 0 0 0 3px rgba(41, 166, 60, 0.3) !important;
}

.stSlider .st-b7 {
    background-color: #29A63C !important;
}

[data-testid="stSliderValue"],
span[data-testid="stSliderValue"] {
    color: #29A63C !important;
    font-weight: bold;
}

/* Progress Bars */
.stProgress>div>div>div>div {
    background-color: #29A63C !important;
}

/* Multiselect tag styling */
.stMultiSelect [data-baseweb="tag"] {
    background-color: #29A63C !important;
    color: #fff !important;
    border: none !important;
}
.stMultiSelect [data-baseweb="tag"] span {
    color: #fff !important;
}
.stMultiSelect [data-baseweb="tag"] svg {
    color: #fff !important;
}

/* Radio Buttons */
[data-baseweb="radio"] label[aria-checked="true"] {
    color: #29A63C !important;
}
[data-baseweb="radio"] div[aria-checked="true"] {
    border-color: #29A63C !important;
    background: #29A63C !important;
}
[data-baseweb="radio"] svg {
    color: #29A63C !important;
}

/* Sticky Sidebar Logo */
.sidebar .stImage>div>div>img {
    position: sticky;
    top: 0;
    background: #111826;
    padding: 1rem 0;
    z-index: 100;
    margin-bottom: 1rem;
}

/* Charts */
.stPlotlyChart {
    --primary-color: #29A63C;
    background-color: #111826;
}
.js-plotly-plot .scatterlayer .trace .lines {
    stroke: #29A63C !important;
}
.js-plotly-plot .barlayer .trace .points path {
    fill: #29A63C !important;
}
.js-plotly-plot .legend .traces .legendtoggle {
    fill: #29A63C !important;
}

/* Metric Cards */
.metric-card {
    background: #1e2533;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(255, 255, 255, 0.05);
    margin-bottom: 20px;
    color: #ffffff;
}

/* Progress Bars in Dashboard */
.progress-container {
    background: #2c3644;
    border-radius: 10px;
    height: 10px;
    margin: 10px 0;
}
.progress-bar {
    height: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #29A63C, #1f8a2c);
    transition: width 0.5s ease;
}

/* Achievement Badge */
.achievement-badge {
    background: #182230;
    border: 1px solid #29A63C;
    border-radius: 15px;
    padding: 10px 15px;
    margin: 5px;
    display: inline-block;
    font-size: 14px;
    color: #ffffff;
}
.achievement-badge.locked {
    opacity: 0.5;
    background: #2c2f36;
}
            
/* Make inactive track neutral dark */
[data-baseweb="slider"] div[role="presentation"] > div:first-child {
    background-color: #2c2f36 !important;
}

/* Make active track (progress bar) green */
[data-baseweb="slider"] div[role="presentation"] > div:first-child > div {
    background-color: #29A63C !important;
}

/* Make slider thumb green */
[data-baseweb="slider"] [role="slider"] {
    background-color: #29A63C !important;
    border-color: #29A63C !important;
}

/* Make active label above the thumb green */
[data-baseweb="slider"] [data-testid="slider-value"] {
    color: #29A63C !important;
    font-weight: 600;
}

/* Ensure all labels/ticks are green */
[data-baseweb="slider"] div[role="presentation"] span {
    color: #29A63C !important;
}

</style>

""", unsafe_allow_html=True)



# Initialize data manager
data_manager = DataManager()

# Calculate metrics
def calculate_savings_rate(income: float, expenses: float) -> float:
    return ((income - expenses) / income) * 100 if income > 0 else 0

def get_monthly_summary() -> dict:
    """Calculate monthly summary including income, expenses, and savings."""
    current_month = datetime.now().strftime('%Y-%m')
    transactions = data_manager.data['transactions']
    
    monthly_transactions = [
        tx for tx in transactions 
        if tx['date'].startswith(current_month)
    ]
    
    income = sum(tx['amount'] for tx in monthly_transactions if tx['amount'] > 0)
    expenses = abs(sum(tx['amount'] for tx in monthly_transactions if tx['amount'] < 0))
    
    return {
        'income': income,
        'expenses': expenses,
        'savings': income - expenses,
        'savings_rate': calculate_savings_rate(income, expenses)
    }

def show_dashboard():
    # Get user data
    monthly_summary = get_monthly_summary()
    
    # Dashboard title and add transaction button
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("📊 Expense Tracker")
    with col2:
        if st.button("💸 Add Transaction", use_container_width=True):
            st.session_state.show_transaction_form = not st.session_state.get('show_transaction_form', False)
    
    # Show transaction form if toggled
    if st.session_state.get('show_transaction_form', False):
        with st.expander("➕ Add New Transaction", expanded=True):
            show_transaction_form()
        st.markdown("---")
    
    # Main metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        net_worth = data_manager.get_net_worth()
        st.markdown(f"<div class='metric-card'><h3>Net Worth</h3><h2>${net_worth:,.2f}</h2></div>", 
                   unsafe_allow_html=True)
    with col2:
        savings = monthly_summary['savings']
        savings_rate = monthly_summary['savings_rate']
        st.markdown(f"<div class='metric-card'><h3>Monthly Savings</h3><h2>${savings:,.2f}</h2>"
                   f"<p>{savings_rate:.1f}% of income</p></div>", unsafe_allow_html=True)
    with col3:
        # Find upcoming bills (expenses in the next 7 days)
        upcoming_bills = [
            tx for tx in data_manager.data['transactions'] 
            if tx['amount'] < 0 and 
            datetime.strptime(tx['date'], '%Y-%m-%d').date() >= datetime.now().date() and
            datetime.strptime(tx['date'], '%Y-%m-%d').date() <= (datetime.now() + timedelta(days=7)).date()
        ]
        
        if upcoming_bills:
            next_bill = min(upcoming_bills, key=lambda x: x['date'])
            days_until = (datetime.strptime(next_bill['date'], '%Y-%m-%d').date() - datetime.now().date()).days
            st.markdown(f"<div class='metric-card'><h3>Next Bill</h3><h2>{next_bill['description']} - ${abs(next_bill['amount']):.2f}</h2>"
                      f"<p>Due in {days_until} days</p></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='metric-card'><h3>No Upcoming Bills</h3><p>You're all set for now!</p></div>", 
                      unsafe_allow_html=True)
    
    # Spending Overview
    st.markdown("### 📊 Monthly Overview")
    
    # Get actual spending data
    spending_by_category = data_manager.get_spending_by_category()
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart of spending by category
        if any(amount > 0 for amount in spending_by_category.values()):
            # Orange color scale for charts
            color_scale = ['#14532D', '#166534', '#15803D', '#16A34A', '#22C55E', '#4ADE80']
            
            fig_pie = px.pie(
                values=list(spending_by_category.values()),
                names=list(spending_by_category.keys()),
                title="Spending by Category",
                hole=0.5,
                color_discrete_sequence=color_scale
            )
            fig_pie.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                hovertemplate='%{label}: $%{value:.2f} (%{percent})<extra></extra>'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No spending data available for this month.")
    
    with col2:
        # Budget vs Actual spending
        budget_data = []
        for category, spent in spending_by_category.items():
            if category in data_manager.data['budgets']:
                budget_data.append({
                    'Category': category,
                    'Amount': spent,
                    'Type': 'Spent',
                    'Budget': data_manager.data['budgets'][category]
                })
        
        if budget_data:
            df_budget = pd.DataFrame(budget_data)
            
            fig = go.Figure()
            
            # Add budget bars
            fig.add_trace(go.Bar(
                x=df_budget['Category'],
                y=df_budget['Budget'],
                name='Budget',
                marker_color='#05F26C',  # Lighter green
                opacity=0.6
            ))
            
            # Add actual spending bars
            fig.add_trace(go.Bar(
                x=df_budget['Category'],
                y=df_budget['Amount'],
                name='Spent',
                marker_color='#14532D',  # Main green
                width=0.4
            ))
            
            fig.update_layout(
                title='Spending vs Budget',
                barmode='group',
                xaxis_title='Category',
                yaxis_title='Amount ($)',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No budget data available.")
    
    # Recent Transactions
    st.markdown("### 📝 Recent Transactions")
    
    # Get recent transactions (last 10)
    if 'transactions' in data_manager.data and data_manager.data['transactions']:
        try:
            # Convert to DataFrame for display
            transactions_df = pd.DataFrame(data_manager.data['transactions'])
            
            # Ensure we have the required columns with fallbacks
            if 'date' not in transactions_df.columns:
                transactions_df['date'] = pd.Timestamp.now().date()
            if 'description' not in transactions_df.columns:
                transactions_df['description'] = 'No description'
            if 'category' not in transactions_df.columns:
                transactions_df['category'] = 'Uncategorized'
                
            # Determine amount handling based on available columns
            if 'amount' in transactions_df.columns and 'type' in transactions_df.columns:
                transactions_df['Amount'] = transactions_df.apply(
                    lambda x: abs(x['amount']) if x['type'] == 'income' else -abs(x['amount']), 
                    axis=1
                )
            elif 'amount' in transactions_df.columns:
                transactions_df['Amount'] = transactions_df['amount']
            else:
                transactions_df['Amount'] = 0.0
            
            # Format for display
            transactions_display = transactions_df[['date', 'description', 'category', 'Amount']].copy()
            transactions_display = transactions_display.rename(columns={
                'date': 'Date',
                'description': 'Description',
                'category': 'Category'
            })
            
            # Sort by date descending (newest first) and get last 10
            transactions_display = transactions_display.sort_values('Date', ascending=False).head(10)
            
        except Exception as e:
            st.error(f"Error processing transactions: {str(e)}")
            return
        
        # Display the transactions
        st.dataframe(
            transactions_display,
            column_config={
                'Amount': st.column_config.NumberColumn(
                    'Amount',
                    help='Transaction amount',
                    format='$%.2f',
                    width='medium'
                ),
                'Date': st.column_config.DateColumn(
                    'Date',
                    help='Transaction date',
                    format='YYYY-MM-DD'
                )
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No recent transactions found.")
    
    # Transaction form is now shown in the expander at the top of the dashboard

def _needs_rag_context(query: str) -> bool:
    """Determine if the query needs RAG context based on keywords."""
    # These are profile-related terms that don't need RAG
    profile_terms = [
        'my', 'i', 'me', 'mine', 'myself',
        'profile', 'account', 'balance', 'transaction', 'spending',
        'savings', 'income', 'expense', 'debt', 'budget', 'goal'
    ]
    
    query_lower = query.lower()
    
    # If the query is clearly about the user's personal data, don't use RAG
    if any(term in query_lower for term in profile_terms):
        return False
        
    # Otherwise, use RAG for general financial knowledge questions
    return True

def get_ai_response(user_input: str, chat_history: list = None) -> str:
    """Get a response from the AI assistant with enhanced financial context.
    
    Args:
        user_input (str): The user's message
        chat_history (list, optional): List of previous messages in the conversation
        
    Returns:
        str: The AI's response
    """
    try:
        # Initialize RAG context as empty
        rag_context = ""
        
        # Only use RAG for general financial knowledge questions
        if _needs_rag_context(user_input):
            try:
                relevant_chunks = rag_service.retrieve_relevant_chunks(user_input, top_k=3)
                if relevant_chunks:
                    rag_context = "\n".join(f"- {chunk[:300]}..." for chunk in relevant_chunks)
                    print(f"Using RAG context for query: {user_input}")
            except Exception as e:
                print(f"Error retrieving RAG context: {e}")
                rag_context = ""
        
        # Get user profile and financial data
        profile = data_manager.get_profile()
        monthly_summary = get_monthly_summary()
        transactions = data_manager.data.get('transactions', [])[:10]  # Get recent transactions
        
        # Prepare chat history context if available
        chat_history_context = ""
        if chat_history and len(chat_history) > 1:  # More than just the current message
            # Only include the last few messages to avoid context window issues
            recent_messages = chat_history[-4:]  # Last 2 exchanges (user + assistant)
            chat_history_context = "\n## CONVERSATION HISTORY\n"
            for msg in recent_messages:
                role = "USER" if msg["role"] == "user" else "ASSISTANT"
                chat_history_context += f"{role}: {msg['content']}\n\n"
        
        # Calculate financial ratios
        monthly_income = profile.get('monthly_income', 5000)
        savings_rate = (monthly_summary['savings'] / monthly_income * 100) if monthly_income > 0 else 0
        
        # Only include financial context if we have meaningful data
        has_financial_data = monthly_income > 0 or len(transactions) > 0
        
        financial_context = {}
        if has_financial_data:
            financial_context = {
                'personal': {
                    'name': profile.get('personal', {}).get('name', 'User'),
                    'age': profile.get('personal', {}).get('age', 'Not specified'),
                },
                'income': {
                    'monthly': monthly_income,
                    'annual': monthly_income * 12 if monthly_income else 0,
                },
                'savings': {
                    'balance': profile.get('assets', {}).get('savings_balance', 0),
                    'rate': round(savings_rate, 1),
                    'emergency_fund': profile.get('assets', {}).get('emergency_fund', 0),
                },
                'debt': {
                    'total': profile.get('debt', {}).get('total_debt', 0),
                    'monthly_payments': profile.get('debt', {}).get('monthly_payments', 0),
                    'types': profile.get('debt', {}).get('debt_types', []),
                },
                'spending': {
                    'monthly': abs(monthly_summary['expenses']),
                    'by_category': {},
                },
                'goals': profile.get('financial_goals', []),
                'risk_tolerance': profile.get('risk_profile', {}).get('risk_tolerance', 'moderate'),
            }
            
            # Calculate spending by category
            for tx in transactions:
                if tx['amount'] < 0:  # Only expenses
                    category = tx.get('category', 'Uncategorized')
                    if category not in financial_context['spending']['by_category']:
                        financial_context['spending']['by_category'][category] = 0
                    financial_context['spending']['by_category'][category] += abs(tx['amount'])
            
            # Generate financial health summary
            financial_health = []
            
            # Emergency fund check
            recommended_emergency_fund = monthly_summary['expenses'] * 3  # 3 months of expenses
            if financial_context['savings']['emergency_fund'] >= recommended_emergency_fund:
                financial_health.append("✅ You have a healthy emergency fund")
            else:
                financial_health.append(f"⚠️ Consider building your emergency fund to cover 3-6 months of expenses (${recommended_emergency_fund:,.2f} recommended)")
            
            # Debt-to-income ratio check
            if monthly_income > 0:
                debt_to_income = (financial_context['debt']['monthly_payments'] / monthly_income) * 100
                if debt_to_income > 36:
                    financial_health.append(f"⚠️ Your debt-to-income ratio is high ({debt_to_income:.1f}%). Consider paying down debt.")
            
            # Savings rate check
            if savings_rate < 20:
                financial_health.append(f"ℹ️ Your savings rate is {savings_rate:.1f}%. Aim for at least 20% for healthy finances.")
            
            # Prepare system message with financial context
            system_message = f"""You are a knowledgeable Financial Advisor having a conversation with a user. You have access to financial knowledge and the user's financial profile.

{chat_history_context if chat_history_context else ''}
            
## FINANCIAL KNOWLEDGE
Here's relevant information from financial resources:
{rag_context if rag_context else 'No relevant financial knowledge found.'}

## USER'S FINANCIAL PROFILE
Here's what I know about the user's financial situation:
- Name: {financial_context['personal']['name']}
- Age: {financial_context['personal']['age']}
- Monthly Income: ${financial_context['income']['monthly']:,.2f}
- Monthly Expenses: ${financial_context['spending']['monthly']:,.2f}
- Savings Rate: {financial_context['savings']['rate']}%
- Debt: ${financial_context['debt']['total']:,.2f}
- Financial Goals: {', '.join(goal['name'] for goal in financial_context['goals']) if financial_context.get('goals') else 'Not specified'}

## INSTRUCTIONS
1. First, address the user's specific question using the financial knowledge provided.
2. If relevant, connect your response to the user's financial situation.
3. Be concise, clear, and actionable in your advice.
4. If the user hasn't provided enough financial information, you can ask clarifying questions."""
        else:
            system_message = """You are a knowledgeable Financial Advisor with expertise in personal finance.

## INSTRUCTIONS
1. Focus on answering the user's question using the financial knowledge provided.
2. Be clear, concise, and practical in your advice.
3. If the question is about their personal finances, you can ask for more details to provide personalized advice."""

        # Add common guidance for both cases
        system_message += """

## ONGOING INTERACTIONS
- Always anchor advice to the stored profile data.  
- Link new questions/goals to current numbers.  
- **Be concise** – short paragraphs or bullets; no filler lines.  
- Each main response should be around 200 words unless the user asks for more detail.

## CORE COMPETENCY AREAS
1. General Principles of Financial Planning  
2. Investment Planning  
3. Insurance Planning  
4. Retirement Savings  
5. Income Planning  
6. Psychology of Financial Planning  

## RESPONSE GUIDELINES
Every advisory answer should include:
1. **Profile Acknowledgment** – reference a relevant number/fact.  
2. **Current Situation Analysis** – what the numbers mean.  
3. **Goal-Specific Strategy** – steps linked to stated goals.  
4. **Implementation Roadmap** – bullet checklist with amounts/percentages.  
5. **Progress Tracking** – how to measure success.  
6. **Actionable Next Step** – the single most useful thing to do now.

### Investment Advice
When discussing investments, include:
- Key risks (2-3 bullet points)
- How it fits the user's profile
- Risk mitigation strategies

## DISCLAIMER
"This analysis is for educational purposes only and not personalized investment advice. Consult a qualified financial professional before making significant decisions."

## TONE & STYLE
Friendly but professional, data-driven, and focused on essentials with bullet points where appropriate."""

        # Generate response using the agent
        response = agent_executor.invoke({
            "input": f"Context:\n{system_message}\n\nUser: {user_input}"
        })
        
        return response.get("output", "I'm sorry, I couldn't generate a response. Could you please rephrase your question?")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"I'm sorry, I encountered an error while processing your request: {str(e)}"

def show_transaction_form(form_key="transaction_form"):
    """Display the transaction form with a unique key."""
    with st.form(key=form_key):
        col1, col2 = st.columns([1, 2])
        with col1:
            transaction_type = st.radio("Type", ["Income", "Expense"], horizontal=True)
        with col2:
            amount = st.number_input("Amount", min_value=0.01, step=0.01, format="%.2f")

        description = st.text_input("Description")

        col1, col2 = st.columns(2)
        with col1:
            category = st.selectbox(
                "Category",
                options=["Food", "Transportation", "Housing", "Entertainment", "Healthcare", "Other"]
            )
        with col2:
            date = st.date_input("Date", value=datetime.now())

        account = st.selectbox("Account", options=["checking", "savings"])  # <-- Add this line

        # Form actions
        col1, col2 = st.columns([1, 1])
        with col1:
            submitted = st.form_submit_button("💾 Save Transaction", use_container_width=True)
        with col2:
            if st.form_submit_button("❌ Cancel", type="secondary", use_container_width=True):
                st.session_state.show_transaction_form = False
                st.rerun()

        if submitted:
            try:
                transaction = {
                    "type": transaction_type.lower(),
                    "amount": float(amount) if transaction_type == "Income" else -float(amount),
                    "description": description,
                    "category": category,
                    "date": date.strftime("%Y-%m-%d"),
                    "account": account  # <-- Ensure this is included
                }
                data_manager.add_transaction(transaction)
                st.success("Transaction saved successfully!")
                st.session_state.show_transaction_form = False
                st.rerun()
            except Exception as e:
                st.error(f"Error saving transaction: {str(e)}")


def show_profile_form():
    """Display the comprehensive profile edit form."""
    # Load existing profile data if available
    profile_data = data_manager.get_profile()
    
    with st.form("profile_form"):
        
        # Personal Information
        st.header("Personal Information")
        col1, col2 = st.columns(2)
        with col1:
            full_name = st.text_input("Full Name", 
                value=profile_data.get("personal", {}).get("name", "John Smith"))
                
            age = st.number_input("Age", 
                min_value=18, 
                max_value=100, 
                value=profile_data.get("personal", {}).get("age", 34), 
                step=1)
                
            marital_status = st.selectbox(
                "Marital Status",
                ["Single", "Married", "Divorced", "Widowed", "Other"],
                index=["Single", "Married", "Divorced", "Widowed", "Other"].index(
                    profile_data.get("personal", {}).get("marital_status", "Married")
                )
            )
            
            dependents = st.number_input(
                "Number of Financial Dependents", 
                min_value=0, 
                value=profile_data.get("personal", {}).get("dependents", 2))
        
        # Employment Information
        st.header("Employment & Income")
        employment_status = st.selectbox(
            "Employment Status",
            ["Employed", "Self-Employed", "Unemployed", "Retired", "Student", "Other"]
        )
        job_title = st.text_input("Job Title/Role", "Software Engineer")
        
        col1, col2 = st.columns(2)
        with col1:
            monthly_income = st.number_input(
                "Monthly Income (after taxes) $", 
                min_value=0, 
                value=6000, 
                step=100
            )
        with col2:
            savings_rate = st.number_input(
                "Monthly Savings Rate (%)", 
                min_value=0, 
                max_value=100, 
                value=15
            )
        
        # Expenses
        st.header("Monthly Expenses")
        col1, col2 = st.columns(2)
        with col1:
            fixed_expenses = st.number_input(
                "Fixed Expenses (rent, bills) $", 
                min_value=0, 
                value=2200
            )
        with col2:
            variable_expenses = st.number_input(
                "Variable Expenses (food, shopping) $", 
                min_value=0, 
                value=800
            )
        
        # Assets
        st.header("Assets")
        col1, col2 = st.columns(2)
        with col1:
            savings_balance = st.number_input(
                "Current Savings (cash, bank) $", 
                min_value=0, 
                value=18000
            )
        with col2:
            investment_value = st.number_input(
                "Total Investment Value $", 
                min_value=0, 
                value=25000
            )
        
        investment_types = st.multiselect(
            "Types of Investments",
            ["Stocks", "Bonds", "Mutual Funds", "ETFs", "Real Estate", "Crypto", "Other"],
            default=["Stocks", "Mutual Funds"]
        )
        
        # Debt
        st.header("Debt")
        col1, col2 = st.columns(2)
        with col1:
            total_debt = st.number_input(
                "Total Debt $", 
                min_value=0, 
                value=12000
            )
        with col2:
            debt_types = st.multiselect(
                "Types of Debt",
                options=["Credit Card", "Mortgage", "Student Loan", "Auto Loan", "Personal Loan", "Other"],
                default=["Credit Card", "Student Loan"]  # Using values that exist in options
            )
        
        # Financial Habits
        st.header("Financial Habits")
        budget_method = st.selectbox(
            "Do you follow a budgeting method? If yes, which one?",
            ["50/30/20 Rule", "Envelope System", "Zero-Based Budgeting", "No specific method"]
        )
        
        track_expenses = st.radio(
            "Do you track your expenses monthly?",
            ["Yes", "No", "Sometimes"]
        )
        
        financial_goals = st.multiselect(
            "What are your financial goals? (Select all that apply)",
            options=["Buy a home", "Save for retirement", "Pay off debt", "Build emergency fund", "Invest in stocks", "Start a business", "Save for education", "Travel"],
            default=["Buy a home", "Save for retirement"]  # Using values that exist in options
        )
        
        # Risk Profile
        st.header("Risk Profile")
        risk_tolerance = st.select_slider(
            "What is your risk tolerance?",
            options=["Very Conservative", "Conservative", "Moderate", "Aggressive", "Very Aggressive"],
            value="Moderate"
        )
        
        investment_experience = st.radio(
            "Have you invested before?",
            ["Yes", "No", "A little"]
        )
        
        investment_horizon = st.select_slider(
            "How long do you want to invest for?",
            options=["Short-term (<1 yr)", "Medium-term (1-5 yrs)", "Long-term (5-10 yrs)", "Very long-term (>10 yrs)"],
            value="Long-term (5-10 yrs)"
        )
        
        investment_style = st.radio(
            "Preferred investment style?",
            ["Passive (set it and forget it)", "Active (I enjoy managing my portfolio)", "Robo-advisor"]
        )
        
        # Insurance
        st.header("Insurance")
        has_insurance = st.radio(
            "Do you have existing insurance?",
            ["Yes", "No", "Some coverage"]
        )
        
        if has_insurance in ["Yes", "Some coverage"]:
            insurance_types = st.multiselect(
                "Which types of insurance do you have?",
                ["Health", "Auto", "Life", "Disability", "Homeowners/Renters", "Other"],
                default=["Health", "Auto", "Life"]
            )
            
            insurance_budget = st.number_input(
                "Monthly insurance premium budget $", 
                min_value=0, 
                value=350
            )
        
        # Retirement Planning
        st.header("Retirement Planning")
        col1, col2 = st.columns(2)
        with col1:
            retirement_age = st.number_input(
                "Target retirement age",
                min_value=30,
                max_value=100,
                value=65,
                step=1
            )
        with col2:
            retirement_savings = st.number_input(
                "Current retirement savings $",
                min_value=0,
                value=30000
            )
        
        monthly_retirement_contribution = st.number_input(
            "Monthly contribution to retirement accounts $",
            min_value=0,
            value=500
        )
        
        retirement_expenses = st.number_input(
            "Expected monthly expenses in retirement $",
            min_value=0,
            value=3000
        )
        
        # Additional Information
        st.header("Additional Information")
        income_changes = st.text_area(
            "Any expected income-related changes in the next year?",
            "Possibly a promotion at work"
        )
        
        money_attitude = st.select_slider(
            "How do you feel about money?",
            options=["Very Stressed", "Somewhat Stressed", "Neutral", "Comfortable", "Very Comfortable"],
            value="Neutral"
        )
        
        impulse_purchases = st.radio(
            "Do you frequently make impulse purchases?",
            ["Never", "Rarely", "Sometimes", "Often"]
        )
        
        money_worries = st.select_slider(
            "How often do you worry about money?",
            options=["Never", "Rarely", "Sometimes", "Often", "Constantly"],
            value="Sometimes"
        )
        
        financial_habits = st.text_area(
            "Which financial habits would you like to improve?",
            "Increase savings rate"
        )
        
        # Form actions
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button(" Save Profile", use_container_width=True):
                # Here you would save all the profile data
                profile_data = {
                    "personal": {
                        "name": full_name,
                        "age": age,
                        "marital_status": marital_status,
                        "dependents": dependents
                    },
                    "employment": {
                        "status": employment_status,
                        "job_title": job_title,
                        "monthly_income": monthly_income
                    },
                    "expenses": {
                        "fixed": fixed_expenses,
                        "variable": variable_expenses,
                        "savings_rate": savings_rate
                    },
                    "assets": {
                        "savings": savings_balance,
                        "investments": {
                            "total_value": investment_value,
                            "types": investment_types
                        }
                    },
                    "debt": {
                        "total": total_debt,
                        "types": debt_types
                    },
                    "risk_profile": {
                        "tolerance": risk_tolerance,
                        "experience": investment_experience,
                        "horizon": investment_horizon,
                        "style": investment_style
                    },
                    "retirement": {
                        "target_age": retirement_age,
                        "current_savings": retirement_savings,
                        "monthly_contribution": monthly_retirement_contribution,
                        "projected_expenses": retirement_expenses
                    },
                    "insurance": {
                        "has_coverage": has_insurance,
                        "types": insurance_types if has_insurance in ["Yes", "Some coverage"] else [],
                        "monthly_budget": insurance_budget if has_insurance in ["Yes", "Some coverage"] else 0
                    },
                    "habits": {
                        "budget_method": budget_method,
                        "tracks_expenses": track_expenses,
                        "goals": financial_goals,
                        "money_attitude": money_attitude,
                        "impulse_purchases": impulse_purchases,
                        "money_worries": money_worries,
                        "improvement_areas": financial_habits
                    }
                }
                
                # Save to data manager
                data_manager.update_profile(profile_data)
                st.success("Profile updated successfully!")
                
        with col2:
            if st.form_submit_button("❌ Cancel", use_container_width=True):
                st.session_state.show_profile_form = False
                st.rerun()

def show_chat():
    """Display the enhanced chat interface with financial assistant."""
    st.title("💬 Finance Assistant")
    
    # Initialize chat messages in session state if not exists
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": "👋 Welcome to FinFluent! I'm your AI-powered financial assistant. I can help you with:\n\n• Analyzing your spending patterns\n• Tracking financial goals\n• Budgeting and saving strategies\n• Debt management\n• Financial planning\n\nTry asking me questions like:""",
                "suggestions": [
                    "What's my current financial health?",
                    "How can I save more money?",
                    "Help me create a budget",
                    "What's my net worth?",
                    "How can I pay off my debt faster?"
                ]
            }
        ]
    
    # Process any pending actions first
    if 'pending_user_message' in st.session_state:
        user_message = st.session_state.pending_user_message
        del st.session_state.pending_user_message
        
        # Add user message to chat history
        st.session_state.chat_messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Get AI response with chat history
        with st.spinner("Analyzing your finances..."):
            response = get_ai_response(
                user_input=user_message,
                chat_history=st.session_state.chat_messages
            )
            
            # Add AI response to chat history
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": response
            })
        
        # Rerun to update the UI
        st.rerun()
    
    # Display chat messages
    for i, message in enumerate(st.session_state.chat_messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display clickable suggestions if available
            if message.get("suggestions") and message["role"] == "assistant" and i == len(st.session_state.chat_messages) - 1:
                cols = st.columns(2)
                for j, suggestion in enumerate(message["suggestions"][:4]):  # Show max 4 suggestions
                    if cols[j % 2].button(suggestion, use_container_width=True, key=f"sug_{i}_{j}"):
                        # Set the pending message and rerun
                        st.session_state.pending_user_message = suggestion
                        st.rerun()
    
    # Chat input
    if prompt := st.chat_input("Ask me about your finances..."):
        st.session_state.pending_user_message = prompt
        st.rerun()
    
    # Add a clear chat button in the sidebar
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.chat_messages = [{
            "role": "assistant",
            "content": "Chat history cleared. How can I help you today?",
            "suggestions": [
                "What's my current financial health?",
                "How can I save more money?",
                "Help me create a budget",
                "What's my net worth?"
            ]
        }]
        st.rerun()

def show_sidebar():
    """Display the sidebar navigation."""
    with st.sidebar:
        # Add logo at the top of the sidebar
        st.image("logo.png", use_container_width=True)
        st.markdown("---")
        
        # Navigation
        st.title("Navigation")
        st.markdown("---")
        if st.button("👤 My Profile", use_container_width=True,
                   type="primary" if st.session_state.get('current_page') == "profile" else "secondary"):
            st.session_state.current_page = "profile"
            st.rerun()
        if st.button("🤖 Financial AI Assistant", use_container_width=True,
                   type="primary" if st.session_state.get('current_page') == "chat" else "secondary"):
            st.session_state.current_page = "chat"
            st.rerun()
        if st.button("📊 Expense Tracker", use_container_width=True, 
                    type="primary" if st.session_state.get('current_page') == "dashboard" else "secondary"):
            st.session_state.current_page = "dashboard"
            st.rerun()

def show_profile():
    """Display the profile form directly for editing."""
    # Welcome message at the top
    user_name = data_manager.data['user'].get('name', 'there')
    st.title(f"👤 Edit Your Profile")
    st.markdown("---")
    
    # Show the profile form directly
    show_profile_form()

def main():
    """Main application function."""
    # Initialize session state with default values
    if 'initialized' not in st.session_state:
        st.session_state.current_page = 'profile'  # Set profile as default page
        st.session_state.chat_messages = []
        st.session_state.show_transaction_form = False
        st.session_state.show_profile_form = False
        st.session_state.initialized = True
    
    # Show the sidebar
    show_sidebar()
    
    # Make sure current_page is set
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'profile'
    
    # Show the appropriate page based on selection
    if st.session_state.current_page == 'dashboard':
        show_dashboard()
    elif st.session_state.current_page == 'chat':
        show_chat()
    else:  # Default to profile
        st.session_state.current_page = 'profile'
        show_profile()
    
    # Show profile form if toggled (shown in main content area)
    if st.session_state.show_profile_form:
        show_profile_form()

if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Please set your OPENAI_API_KEY in the .env file")
    else:
        main()
