import asyncio
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, TypedDict, Annotated, Sequence, Union
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool
from langchain.memory import ConversationBufferWindowMemory
from pydantic import BaseModel, Field
import dotenv
from data_manager import DataManager
import os
from datetime import datetime, timedelta
import httpx
import json
from typing import Optional, Dict, Any

# Load environment variables
dotenv.load_dotenv()

# Initialize data manager
data_manager = DataManager()

@tool
def get_account_balance(account_name: str = None) -> str:
    """Fetch the current balance for a given bank account.
    
    Args:
        account_name: The name of the account (e.g., 'checking', 'savings'). If not provided, returns all accounts.
    """
    try:
        if account_name:
            # Get specific account balance
            if account_name.lower() not in data_manager.data['accounts']:
                return f"Error: No account found with name '{account_name}'. Available accounts: {', '.join(data_manager.data['accounts'].keys())}"
            
            balance = data_manager.data['accounts'][account_name.lower()]
            
            # Get recent transactions for this account
            recent_tx = [
                tx for tx in data_manager.data['transactions']
                if tx['account'].lower() == account_name.lower()
            ][-5:]  # Last 5 transactions
            
            response = f"{account_name.capitalize()} Account Balance: ${balance:,.2f}\n\n"
            if recent_tx:
                response += "Recent Transactions:\n"
                for tx in sorted(recent_tx, key=lambda x: x['date'], reverse=True):
                    amount = f"-${abs(tx['amount']):.2f}" if tx['amount'] < 0 else f"${tx['amount']:.2f}"
                    response += f"  {tx['date']}: {tx['description']} ({tx['category']}) {amount}\n"
            else:
                response += "No recent transactions.\n"
                
            return response
        else:
            # Return all accounts
            response = "Account Balances:\n"
            for account, balance in data_manager.data['accounts'].items():
                response += f"- {account.capitalize()}: ${balance:,.2f}\n"
            return response
            
    except Exception as e:
        return f"Error retrieving account balance: {str(e)}"

@tool
def analyze_spending(timeframe: str = "current_month", category: str = None) -> str:
    """Analyze spending patterns over a specified timeframe.
    
    Args:
        timeframe: The timeframe to analyze ('current_month', 'last_month', 'last_30_days', 'all_time')
        category: Optional category to filter by
    """
    try:
        # Get all transactions
        transactions = data_manager.data['transactions']
        
        if not transactions:
            return "No transaction data available."
        
        # Filter by timeframe
        now = datetime.now()
        if timeframe == 'current_month':
            current_month = now.strftime('%Y-%m')
            transactions = [tx for tx in transactions if tx['date'].startswith(current_month)]
        elif timeframe == 'last_month':
            last_month = (now.replace(day=1) - timedelta(days=1)).strftime('%Y-%m')
            transactions = [tx for tx in transactions if tx['date'].startswith(last_month)]
        elif timeframe == 'last_30_days':
            thirty_days_ago = (now - timedelta(days=30)).strftime('%Y-%m-%d')
            transactions = [tx for tx in transactions if tx['date'] >= thirty_days_ago]
        # 'all_time' uses all transactions
        
        # Filter by category if specified
        if category:
            transactions = [tx for tx in transactions if tx['category'].lower() == category.lower()]
        
        # Calculate spending by category (only expenses, not income)
        spending_data = {}
        for tx in transactions:
            if tx['amount'] < 0:  # Only count expenses
                cat = tx['category']
                spending_data[cat] = spending_data.get(cat, 0) + abs(tx['amount'])
        
        if not spending_data:
            return f"No spending data found for the specified timeframe: {timeframe}"
        
        # Calculate total and percentages
        total = sum(spending_data.values())
        analysis = f"Spending Analysis ({timeframe.replace('_', ' ').title()})\n"
        if category:
            analysis += f"Category: {category}\n"
        analysis += f"Total Spending: ${total:,.2f}\n\n"
        
        # Sort categories by spending (highest first)
        sorted_spending = sorted(spending_data.items(), key=lambda x: x[1], reverse=True)
        
        for category, amount in sorted_spending:
            percentage = (amount / total) * 100 if total > 0 else 0
            analysis += f"- {category}: ${amount:,.2f} ({percentage:.1f}%)\n"
        
        # Add insights
        analysis += "\nInsights:\n"
        top_category = max(spending_data.items(), key=lambda x: x[1]) if spending_data else (None, 0)
        
        if top_category[0]:
            analysis += f"- Your highest spending category is {top_category[0]} (${top_category[1]:,.2f})\n"
            
            # Check against budget if available
            if top_category[0] in data_manager.data['budgets']:
                budget = data_manager.data['budgets'][top_category[0]]
                if top_category[1] > budget * 0.9:  # If spent > 90% of budget
                    analysis += f"  - You've spent {top_category[1]/budget*100:.1f}% of your ${budget:,.2f} budget for {top_category[0]}.\n"
                    if top_category[1] > budget:
                        analysis += "  - You've exceeded your budget in this category.\n"
                    else:
                        analysis += "  - You're close to your budget limit.\n"
        
        # Check for any unusual spending patterns
        if len(transactions) > 10:  # Only if we have enough data
            avg_daily_spend = total / 30  # Rough average
            daily_spending = {}
            for tx in transactions:
                if tx['amount'] < 0:  # Only expenses
                    day = tx['date']
                    daily_spending[day] = daily_spending.get(day, 0) + abs(tx['amount'])
            
            if daily_spending:
                avg_recent = sum(daily_spending.values()) / len(daily_spending)
                if avg_recent > avg_daily_spend * 1.5:
                    analysis += "\n⚠️ Your recent daily spending is higher than average. Consider reviewing your expenses.\n"
        
        return analysis
        
    except Exception as e:
        return f"Error analyzing spending: {str(e)}"

# Investment suggestions have been removed - using simpler financial planning approach now

@tool
def get_net_worth() -> str:
    """Calculate the user's current net worth."""
    try:
        # Sum all account balances
        total_assets = sum(data_manager.data['accounts'].values())
        
        # Add other assets if available
        if 'assets' in data_manager.data:
            total_assets += sum(asset.get('value', 0) for asset in data_manager.data['assets'])
        
        # Calculate liabilities (debt)
        total_liabilities = data_manager.get_profile().get('total_debt', 0)
        
        net_worth = total_assets - total_liabilities
        
        # Generate response with emojis and formatting
        response = "## 💰 Net Worth Summary\n\n"
        response += f"**Total Assets:** ${total_assets:,.2f}\n"
        response += f"**Total Liabilities:** ${total_liabilities:,.2f}\n"
        response += f"**Net Worth:** ${net_worth:,.2f}\n\n"
        
        # Add some insights
        if net_worth > 0:
            response += "✅ Your net worth is positive! You own more than you owe."
        else:
            response += "⚠️ Your net worth is negative. Consider focusing on paying down debt."
            
        return response
    except Exception as e:
        return f"Error calculating net worth: {str(e)}"

@tool
def get_financial_goals() -> str:
    """Get the user's current financial goals and progress."""
    try:
        goals = data_manager.data.get('goals', [])
        if not goals:
            return "You haven't set any financial goals yet. Consider setting some to better plan your finances."
        
        response = "## 🎯 Your Financial Goals\n\n"
        
        for i, goal in enumerate(goals, 1):
            target = goal.get('target', 0)
            saved = goal.get('saved', 0)
            progress = (saved / target * 100) if target > 0 else 0
            
            response += f"### {i}. {goal.get('name', 'Unnamed Goal')}\n"
            response += f"- **Target:** ${target:,.2f}\n"
            response += f"- **Saved:** ${saved:,.2f}\n"
            response += f"- **Progress:** {progress:.1f}%\n"
            
            if 'target_date' in goal:
                target_date = datetime.strptime(goal['target_date'], '%Y-%m-%d').date()
                today = datetime.now().date()
                days_remaining = (target_date - today).days
                response += f"- **Target Date:** {goal['target_date']} ({days_remaining} days remaining)\n"
                
                # Calculate required savings
                if days_remaining > 0 and progress < 100:
                    remaining = target - saved
                    monthly_savings_needed = remaining / (days_remaining / 30.44)  # Average days per month
                    response += f"- **Monthly Savings Needed:** ${monthly_savings_needed:,.2f}/month\n"
            
            response += "\n"
        
        return response
    except Exception as e:
        return f"Error retrieving financial goals: {str(e)}"

class PerplexitySearch:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY environment variable not set")
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "accept": "application/json"
        }
        self.timeout = 30.0
    
    def search(self, query: str, max_results: int = 3) -> str:
        """Search the web using Perplexity API."""
        try:
            print(f"🔍 Performing search for: {query}")
            
            # Create a more specific prompt for financial news
            prompt = f"""You are a financial news assistant. Provide a concise summary of the most important 
            financial news and market updates as of {datetime.now().strftime('%B %d, %Y')}. 
            Focus on major market movements, economic indicators, and significant corporate news.
            Include relevant numbers and statistics where available.
            
            Format your response with clear sections and bullet points.
            Always cite your sources at the end.
            
            Current query: {query}"""
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant that provides accurate and concise information."},
                {"role": "user", "content": prompt}
            ]
            
            data = {
                "model": "sonar-medium-online",
                "messages": messages,
                "max_tokens": 2000,  # Increased for more detailed responses
                "temperature": 0.3,   # Slightly higher for more varied responses
                "top_p": 0.95
            }
            
            print(f"📡 Sending request to Perplexity API...")
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.base_url,
                    headers=self.headers,
                    json=data,
                    timeout=self.timeout
                )
                
                print(f"🔔 Response status: {response.status_code}")
                if response.status_code != 200:
                    print(f"❌ Error response: {response.text}")
                    return f"Error: Received status code {response.status_code} from Perplexity API. Please try again later."
                
                result = response.json()
                print("📦 Received response from Perplexity API")
                
                # Debug: Print the full response structure
                print("🔍 Response structure:", json.dumps(result, indent=2)[:500] + "...")
                
                # Extract the assistant's response
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message'].get('content', 'No content')
                    print(f"✅ Successfully extracted response ({len(content)} characters)")
                    return content
                
                return "I couldn't find any relevant information. The response format was unexpected."
                
        except httpx.TimeoutException:
            return "Request timed out. The server took too long to respond. Please try again later."
        except Exception as e:
            import traceback
            error_msg = f"Error performing search: {str(e)}\n\n{traceback.format_exc()}"
            print(f"❌ {error_msg}")
            return "I encountered an error while trying to fetch the latest information. Please try again in a few moments."

# Simple web search function
def web_search(query: str) -> str:
    """Perform a web search using Perplexity API."""
    try:
        print(f"🔍 Web search query: {query}")
        
        # Direct API call to Perplexity
        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        # Create a more specific prompt
        prompt = f"""You are a helpful financial assistant. Answer the following question with 
        up-to-date information. Include specific details, numbers, and sources where possible.
        
        Question: {query}
        
        Please provide a detailed response with:
        1. A clear answer to the question
        2. Relevant statistics or data points
        3. Sources of information
        """
        
        data = {
            "model": "sonar-medium-online",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2000,
            "temperature": 0.3
        }
        
        print("📡 Sending request to Perplexity API...")
        response = httpx.post(url, headers=headers, json=data, timeout=30.0)
        response.raise_for_status()
        
        result = response.json()
        print("✅ Received response from Perplexity API")
        
        if 'choices' in result and result['choices']:
            return result['choices'][0]['message']['content']
            
        return "I couldn't find any relevant information for your query."
        
    except Exception as e:
        error_msg = f"Web search error: {str(e)}"
        print(f"❌ {error_msg}")
        return "I'm having trouble fetching that information right now. Please try again later."

# List of tools to be used by the personal finance agent
tools = [
    Tool(
        name=get_account_balance.name,
        func=get_account_balance,
        description=get_account_balance.description
    ),
    Tool(
        name=analyze_spending.name,
        func=analyze_spending,
        description=analyze_spending.description
    ),
    Tool(
        name=get_net_worth.name,
        func=get_net_worth,
        description=get_net_worth.description
    ),
    Tool(
        name=get_financial_goals.name,
        func=get_financial_goals,
        description=get_financial_goals.description
    )
    # Note: web_search is handled separately and not included in the tools list
]

print("🛠️  Registered tools:", [tool.name for tool in tools])

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Define the system message
system_message = """You are a helpful, knowledgeable, and friendly personal finance advisor AI. Your goal is to assist users with their financial questions and provide clear, actionable advice.

You have access to the following tools:
- get_account_balance: Check bank balances. Always ask for the bank_id if not provided.
- analyze_spending: Categorize expenses over a specific timeframe.
- get_net_worth: Calculate the user's total net worth including assets and liabilities.
- get_financial_goals: Review progress towards financial goals and savings targets.
- web_search: Search the web for current information using Perplexity's API. Use this for any questions about recent events, news, or information that might be outside your training data. Always cite your sources when using this tool.

Guidelines:
1. Be conversational but professional.
2. Explain financial terms in simple language.
3. Format numbers properly (e.g., $1,234.56).
4. If a question is unclear, ask for clarification.
5. For investment advice, always consider the user's risk tolerance.
6. When analyzing spending, provide insights and suggestions for improvement.
7. Help users understand their financial health by discussing net worth and goal progress.
8. If you don't know something, be honest and say so.

Example interactions:
User: What's my net worth?
You: I can calculate your net worth by looking at all your assets and liabilities. Would you like me to do that now?

User: How am I doing with my financial goals?
You: I can review your progress towards your financial goals. Would you like me to check that for you?

User: Where is my money going?
You: I can analyze your spending patterns. Would you like to see your spending for the last month, or a different time period?

User: Should I invest in stocks?
You: I can suggest investments based on your risk tolerance. Would you describe your risk tolerance as low, medium, or high?

User: What is the current inflation rate?
You: Let me find the most recent inflation data for you. (Uses web_search tool)

User: What's happening in the stock market today?
You: I'll check the latest market updates. (Uses web_search tool)
"""

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    MessagesPlaceholder(variable_name="chat_history"),
])

# Initialize the language model with better parameters
llm = ChatOpenAI(
    model="gpt-4-turbo",  # Using GPT-4 for better reasoning and tool use
    temperature=0.2,  # Slight randomness for more natural responses
    max_tokens=2000,  # Allow for more detailed responses
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize conversation memory
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=10,  # Keep last 10 messages in memory
    return_messages=True
)

# Create the agent with memory and tools
agent = create_tool_calling_agent(llm, tools, prompt)

# Create the agent executor with better configuration
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,  # Better error handling
    max_iterations=5,            # Prevent very long chains of thought
    return_intermediate_steps=True,  # Better for debugging
    memory=memory,              # Add memory to the executor
    early_stopping_method="generate"  # Better handling of tool calls
)

def should_use_web_search(query: str) -> bool:
    """Determine if a query should use web search."""
    # First, check if it's a personal finance query
    personal_finance_terms = [
        'my account', 'my balance', 'my spending', 'my transactions',
        'my budget', 'my savings', 'my expenses', 'my income',
        'net worth', 'financial goals', 'spending report'
    ]
    
    query_lower = query.lower()
    
    # Only use personal finance agent for very specific personal queries
    if any(term in query_lower for term in personal_finance_terms):
        print(f"💼 Using personal finance agent for query: {query}")
        return False
        
    # For all other queries, use web search
    print(f"🌐 Using web search for query: {query}")
    return True

def get_ai_response(user_input: str) -> str:
    """Get AI response, using web search or personal finance agent as needed."""
    try:
        # First try personal finance agent for specific queries
        if not should_use_web_search(user_input):
            try:
                print("💼 Trying personal finance agent...")
                response = agent_executor.invoke({
                    "input": user_input, 
                    "chat_history": memory.chat_memory.messages
                })
                if response and "output" in response:
                    return response["output"]
            except Exception as e:
                print(f"⚠️ Personal finance agent failed, falling back to web search: {str(e)}")
        
        # Default to web search for everything else
        print("🌐 Using web search...")
        return web_search(user_input)
        
    except Exception as e:
        import traceback
        error_msg = f"Error in get_ai_response: {str(e)}\n\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        return "I encountered an error while processing your request. Please try again."

def display_welcome_message():
    """Display a welcome message with example commands."""
    print("\n" + "="*60)
    print("🌟 Welcome to your Personal Finance Advisor! 🌟".center(60))
    print("="*60)
    print("\nI can help you with:")
    print("  • Viewing account balances")
    print("  • Analyzing spending patterns")
    print("  • Tracking financial goals")
    print("  • Investment recommendations")
    print("  • Calculating net worth")
    print("\nExample questions:")
    print("  • What's my current bank balance?")
    print("  • Show my net worth")
    print("  • Analyze my spending for the last month")
    print("  • How am I doing with my financial goals?")
    print("  • Suggest investments for medium risk tolerance")
    print("\nType 'exit' or 'quit' to end the session.")
    print("-"*60 + "\n")

def format_response(text):
    """Format the response text for better readability in the console."""
    # Simple formatting for markdown-like syntax
    text = text.replace('**', '').replace('*', '•')
    return text

def main():
    # Display welcome message
    display_welcome_message()
    
    # Initialize chat history
    chat_history = []
    
    try:
        while True:
            try:
                # Get user input
                user_input = input("\n💬 You: ").strip()
                
                if user_input.lower() in ['exit', 'quit']:
                    print("\n👋 Thank you for using the Personal Finance Advisor. Goodbye!")
                    break
                
                if not user_input:
                    continue
                    
                print("\n🤔 Thinking...")
                
                # Get AI response using our custom function
                try:
                    full_response = get_ai_response(user_input)
                except Exception as e:
                    full_response = f"An error occurred: {str(e)}"
                
                # Update chat history
                chat_history.extend([
                    HumanMessage(content=user_input),
                    AIMessage(content=full_response)
                ])
                
                # Print the formatted response
                print("\n" + "-"*60)
                print("\n💡 Assistant:")
                print(format_response(response_text))
                print("\n" + "-"*60)
                
            except KeyboardInterrupt:
                print("\n👋 Thank you for using the Personal Finance Advisor. Goodbye!")
                break
            except Exception as e:
                error_msg = f"I'm sorry, I encountered an error: {str(e)}. Please try again or rephrase your question."
                print("\n⚠️", error_msg)
                chat_history.append(AIMessage(content=error_msg))
                
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {str(e)}")
        print("Please restart the application.")
    finally:
        print("\nSession ended.")

if __name__ == "__main__":
    main()
