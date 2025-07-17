from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, EmailStr
from typing import List, Dict, Optional, Union
from passlib.context import CryptContext
from datetime import date, datetime, timedelta
import jwt
from jwt.exceptions import PyJWTError
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd
import os
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from data_access import DatabaseFactory, DatabaseInterface

# Load configuration
from dotenv import load_dotenv
load_dotenv()

# Constants
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "expense_tracker_db"
SECRET_KEY = "YOUR_SECRET_KEY"  # Replace with a secure key in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Models (keep all your existing models here)
# ... [All your existing models remain unchanged] ...

# Application setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database
    app.db = DatabaseFactory.create("mongodb", MONGO_URI, DB_NAME)
    await app.db.initialize()
    yield
    # Cleanup
    await app.db.close()

app = FastAPI(title="Expense Tracker API", description="API for tracking personal expenses", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper functions (keep all your existing helper functions)
# ... [All your existing helper functions remain unchanged] ...

# Expense Tracker class now uses the database interface
class ExpenseTracker:
    def __init__(self, db: DatabaseInterface, user_id: str):
        self.db = db
        self.user_id = user_id
    
    async def create_account(self, name: str, type: str, initial_balance: float = 0.0):
        account_data = {
            "name": name,
            "type": type,
        }
        try:
            account_id = await self.db.create_account(self.user_id, account_data)
            if initial_balance != 0:
                trans_type = "credit" if initial_balance > 0 else "debit"
                await self.add_transaction(
                    date=date.today(),
                    description="Opening Balance",
                    place="Opening Balance",
                    amount=abs(initial_balance),
                    transaction_type=trans_type,
                    category="Initial Balance",
                    account_name=name
                )
            return account_id
        except Exception as e:
            raise ValueError(str(e))
    
    async def add_transaction(self, date: date, description, place, amount, transaction_type, 
                       category, account_name, to_account="None", paid_by="Self", status="Pending"):
        transaction_data = {
            'date': datetime.combine(date, datetime.min.time()),
            'description': description,
            'place': place,
            'amount': amount,
            'type': transaction_type,
            'category': category,
            'account': account_name,
            'to_account': to_account,
            'paid_by': paid_by,
            'status': status,
        }
        try:
            return await self.db.add_transaction(self.user_id, transaction_data)
        except Exception as e:
            raise ValueError(str(e))
    
    async def delete_transaction(self, transaction_id: str) -> bool:
        return await self.db.delete_transaction(self.user_id, transaction_id)
    
    async def update_transaction_status(self, transaction_id: str, new_status: str) -> bool:
        return await self.db.update_transaction_status(self.user_id, transaction_id, new_status)
    
    async def get_transactions(self, limit: Optional[int] = None) -> List[dict]:
        transactions = await self.db.get_transactions(self.user_id)
        return transactions[:limit] if limit else transactions
    
    async def get_transactions_by_filter(self, filters: dict) -> List[dict]:
        # Convert the filter payload to database query format
        query_filter = {}
        
        if filters.get("searchTerm"):
            query_filter["$or"] = [
                {"description": {"$regex": filters["searchTerm"], "$options": "i"}},
                {"place": {"$regex": filters["searchTerm"], "$options": "i"}},
                {"category": {"$regex": filters["searchTerm"], "$options": "i"}},
            ]
        
        date_filter = {}
        if filters.get("dateFrom"):
            date_filter["$gte"] = datetime.combine(filters["dateFrom"], datetime.min.time())
        if filters.get("dateTo"):
            date_filter["$lte"] = datetime.combine(filters["dateTo"], datetime.max.time())
        if date_filter:
            query_filter["date"] = date_filter
        
        if filters.get("type"):
            query_filter["type"] = {"$in": filters["type"]}
        if filters.get("categories"):
            query_filter["category"] = {"$in": filters["categories"]}
        if filters.get("accounts"):
            query_filter["account"] = {"$in": filters["accounts"]}
        
        amount_filter = {}
        if filters.get("minAmount") is not None:
            amount_filter["$gte"] = filters["minAmount"]
        if filters.get("maxAmount") is not None:
            amount_filter["$lte"] = filters["maxAmount"]
        if amount_filter:
            query_filter["amount"] = amount_filter
        
        return await self.db.get_transactions(self.user_id, query_filter)
    
    async def get_all_account_balances(self) -> Dict[str, float]:
        return await self.db.get_account_balances(self.user_id)
    
    async def get_spending_by_category(self, start_date: Optional[date] = None, end_date: Optional[date] = None) -> Dict[str, float]:
        transactions = await self.get_transactions()
        if not transactions:
            return {}
        
        df = pd.DataFrame(transactions)
        df['date'] = pd.to_datetime(df['date'])
        
        if start_date and end_date:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
        
        if not df.empty:
            debit_df = df[df['type'].str.lower() == 'debit']
            if not debit_df.empty:
                return debit_df.groupby('category')['amount'].sum().to_dict()
        return {}
    
    async def plot_spending_by_category(self, start_date: Optional[date] = None, end_date: Optional[date] = None) -> Optional[str]:
        category_spending = await self.get_spending_by_category(start_date, end_date)
        if not category_spending:
            return None
        
        plt.figure(figsize=(10, 6))
        plt.bar(category_spending.keys(), category_spending.values())
        plt.xlabel('Category')
        plt.ylabel('Amount Spent')
        plt.title('Spending by Category')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    async def plot_monthly_spending(self) -> Optional[str]:
        transactions = await self.get_transactions()
        if not transactions:
            return None
        
        df = pd.DataFrame(transactions)
        df['date'] = pd.to_datetime(df['date'])
        
        if not df.empty:
            debit_df = df[df['type'].str.lower() == 'debit']
            if not debit_df.empty:
                debit_df['Month'] = debit_df['date'].dt.strftime('%Y-%m')
                monthly_spending = debit_df.groupby('Month')['amount'].sum()
                
                plt.figure(figsize=(12, 6))
                monthly_spending.plot(kind='bar')
                plt.xlabel('Month')
                plt.ylabel('Amount Spent')
                plt.title('Monthly Spending')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                plt.close()
                
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
        return None

# Dependency to get tracker instance
async def get_tracker(user: dict) -> ExpenseTracker:
    return ExpenseTracker(app.db, str(user["_id"]))

# All your existing API endpoints remain the same, just change how they get the tracker instance
# ... [All your existing endpoints remain unchanged] ...

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)