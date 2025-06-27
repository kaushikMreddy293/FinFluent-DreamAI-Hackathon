import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, TypedDict, Union
from pydantic import BaseModel, Field

# Model Definitions
class UserProfile(BaseModel):
    """User profile model for financial data."""
    full_name: str = ""
    age: Optional[int] = None
    employment_status: str = ""
    job_title: str = ""
    marital_status: str = ""
    dependents: int = 0
    risk_tolerance: str = "moderate"  # low, moderate, high
    # Investment preferences removed - using simpler model now
    financial_goals: List[str] = []
    retirement_age: int = 65
    monthly_income: float = 10000.0
    monthly_expenses: float = 0.0

class Transaction(BaseModel):
    """Financial transaction model."""
    id: int
    date: str
    amount: float
    category: str
    description: str
    account: str

class Goal(BaseModel):
    """Financial goal model."""
    id: Optional[int] = None
    name: str = ""
    target: float = 0.0
    saved: float = 0.0
    target_date: Optional[str] = None
    category: str = ""

# Investment portfolio model removed - using simpler model now

class DataManager:
    def __init__(self, data_file: str = 'finance_data.json'):
        self.data_file = data_file
        self.data = self._load_data()
    
    def _load_data(self) -> Dict[str, Any]:
        """Load data from JSON file or create default structure."""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    # Ensure all required top-level keys exist
                    data.setdefault('transactions', [])
                    data.setdefault('accounts', {'checking': 1000.00, 'savings': 5000.00})
                    data.setdefault('budgets', {
                        'Housing': 1200.00,
                        'Food': 500.00,
                        'Transportation': 200.00,
                        'Entertainment': 100.00,
                        'Bills': 300.00,
                        'Other': 200.00
                    })
                    data.setdefault('goals', [])
                    # Investments data structure removed - using simpler model now
                    data.setdefault('user_profile', UserProfile().model_dump())
                    return data
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error loading data: {e}")
                
        # Return default data structure
        return {
            'transactions': [],
            'accounts': {
                'checking': 1000.00,
                'savings': 5000.00
            },
            'budgets': {
                'Housing': 1200.00,
                'Food': 500.00,
                'Transportation': 200.00,
                'Entertainment': 100.00,
                'Bills': 300.00,
                'Other': 200.00
            },
            'goals': [],
            'investments': {
                'stocks': 0.0,
                'bonds': 0.0,
                'etfs': 0.0,
                'mutual_funds': 0.0,
                'crypto': 0.0,
                'real_estate': 0.0
            },
            'user_profile': UserProfile().model_dump()
        }
    
    def save_data(self) -> None:
        """Save data to the JSON file."""
        try:
            with open(self.data_file, 'w') as f:
                # Convert any Pydantic models to dict before serialization
                serializable_data = {}
                for key, value in self.data.items():
                    if hasattr(value, 'model_dump'):
                        serializable_data[key] = value.model_dump()
                    elif isinstance(value, dict):
                        serializable_data[key] = value
                    else:
                        serializable_data[key] = value
                json.dump(serializable_data, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving data: {e}")
    
    # User Profile Methods
    def update_profile(self, profile_data: Dict[str, Any]) -> bool:
        """Update user profile with new data."""
        try:
            # Validate profile data against the UserProfile model
            updated_profile = UserProfile(**{**self.data['user_profile'], **profile_data})
            self.data['user_profile'] = updated_profile.model_dump()
            self.save_data()
            return True
        except Exception as e:
            print(f"Error updating profile: {e}")
            return False
    
    def get_profile(self) -> Dict[str, Any]:
        """Get the complete user profile."""
        return self.data.get('user_profile', {})
    
    # Transaction Methods
    def add_transaction(self, transaction_data: Dict[str, Any]) -> Optional[int]:
        """Add a new transaction and update relevant accounts."""
        try:
            # Set default values
            transaction_data.setdefault('id', len(self.data['transactions']) + 1)
            transaction_data.setdefault('date', datetime.now().strftime('%Y-%m-%d'))
            
            # Validate transaction data
            transaction = Transaction(**transaction_data)
            
            # Update transactions
            self.data['transactions'].append(transaction.model_dump())
            
            # Update account balance
            account = transaction_data.get('account', 'checking')
            if account in self.data['accounts']:
                self.data['accounts'][account] = round(
                    self.data['accounts'][account] + transaction_data['amount'], 2
                )
            else:
                self.data['accounts'][account] = transaction_data['amount']
            
            self.save_data()
            return transaction.id
        except Exception as e:
            print(f"Error adding transaction: {e}")
            return None
    
    def get_recent_transactions(self, limit: int = 5) -> List[Dict]:
        """Get most recent transactions."""
        try:
            sorted_transactions = sorted(
                self.data['transactions'],
                key=lambda x: x.get('date', ''),
                reverse=True
            )
            return sorted_transactions[:limit]
        except Exception as e:
            print(f"Error getting transactions: {e}")
            return []
    
    # Account Methods
    def get_balance(self, account: Optional[str] = None) -> float:
        """Get balance for a specific account or total balance."""
        try:
            if account:
                return round(self.data['accounts'].get(account, 0), 2)
            return round(sum(self.data['accounts'].values()), 2)
        except Exception as e:
            print(f"Error getting balance: {e}")
            return 0.0
    
    # Budget Methods
    def get_spending_by_category(self, days: int = 30) -> Dict[str, float]:
        """Get spending by category for the given number of days."""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            spending = {}
            
            for t in self.data['transactions']:
                if t.get('amount', 0) < 0 and t.get('date', '') >= cutoff_date:
                    category = t.get('category', 'Uncategorized')
                    spending[category] = round(spending.get(category, 0) + abs(t['amount']), 2)
                    
            return spending
        except Exception as e:
            print(f"Error getting spending by category: {e}")
            return {}
    
    # Investment methods removed - using simpler model now
    
    # Goal Methods
    def get_goals(self) -> List[Dict]:
        """Get all financial goals."""
        return self.data.get('goals', [])
    
    def add_goal(self, goal_data: Dict[str, Any]) -> Optional[int]:
        """Add a new financial goal."""
        try:
            goal_data.setdefault('id', len(self.data['goals']) + 1)
            goal = Goal(**goal_data)
            self.data['goals'].append(goal.model_dump())
            self.save_data()
            return goal.id
        except Exception as e:
            print(f"Error adding goal: {e}")
            return None
    
    def update_goal_progress(self, goal_id: int, amount: float) -> bool:
        """Update progress towards a financial goal."""
        try:
            for goal in self.data['goals']:
                if goal.get('id') == goal_id:
                    goal['saved'] = round(float(amount), 2)
                    self.save_data()
                    return True
            return False
        except Exception as e:
            print(f"Error updating goal progress: {e}")
            return False
    
    # Analysis Methods
    def get_net_worth(self) -> float:
        """Calculate total net worth."""
        try:
            cash_assets = sum(self.data['accounts'].values())
            investment_assets = sum(self.data.get('investments', {}).values())
            total_debt = float(self.data.get('user_profile', {}).get('total_debt', 0))
            return round(cash_assets + investment_assets - total_debt, 2)
        except Exception as e:
            print(f"Error calculating net worth: {e}")
            return 0.0
    
    def get_monthly_summary(self) -> Dict[str, float]:
        """Get summary of income and expenses for the current month."""
        try:
            current_month = datetime.now().strftime('%Y-%m')
            monthly_data = {'income': 10000.0, 'expenses': 0.0}
            
            for t in self.data['transactions']:
                if t.get('date', '').startswith(current_month):
                    amount = float(t.get('amount', 0))
                    if amount > 0:
                        monthly_data['income'] = round(monthly_data['income'] + amount, 2)
                    else:
                        monthly_data['expenses'] = round(monthly_data['expenses'] + abs(amount), 2)
            
            return monthly_data
        except Exception as e:
            print(f"Error getting monthly summary: {e}")
            return {'income': 0.0, 'expenses': 0.0}