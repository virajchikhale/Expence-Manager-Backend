from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import os
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime
import json

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Expense Tracker API", description="API for tracking personal expenses")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Pydantic models for request/response validation
class TransactionBase(BaseModel):
    date: str
    description: str
    name: str
    amount: float
    type: str
    category: str
    account: str
    to_account: Optional[str] = "None"
    paid_by: Optional[str] = "Self"
    status: Optional[str] = "Pending"

class TransactionCreate(TransactionBase):
    pass

class TransactionUpdate(BaseModel):
    status: str

class TransactionResponse(TransactionBase):
    index: int

class SpendingResponse(BaseModel):
    category: str
    amount: float

class ChartResponse(BaseModel):
    chart: str  # Base64 encoded image

class ErrorResponse(BaseModel):
    success: bool = False
    error: str

class SuccessResponse(BaseModel):
    success: bool = True
    message: str

class BalanceResponse(BaseModel):
    success: bool = True
    balances: Dict[str, float]

class SpendingCategoryResponse(BaseModel):
    success: bool = True
    spending: Dict[str, float]

class ChartResponseModel(BaseModel):
    success: bool = True
    chart: str  # Base64 encoded string

class TransactionListResponse(BaseModel):
    success: bool = True
    transactions: List[dict]


class ExpenseTracker:
    def __init__(self):
        self.transactions = []
        self.columns = [
            'date', 'description', 'name', 'amount', 'type', 
            'category', 'account', 'to_account', 'paid_by', 'status'
        ]
    
    def add_transaction(self, date, description, name, amount, transaction_type, 
                       category, account_name, to_account="None", paid_by="Self", status="Pending"):
        transaction = {
            'date': date,
            'description': description,
            'name': name,
            'amount': amount,
            'type': transaction_type,
            'category': category,
            'account': account_name,
            'to_account': to_account,
            'paid_by': paid_by,
            'status': status
        }
        self.transactions.append(transaction)
        return len(self.transactions) - 1  # Return index of new transaction
    
    def delete_transaction(self, index):
        if 0 <= index < len(self.transactions):
            del self.transactions[index]
            return True
        return False
    
    def update_transaction_status(self, index, new_status):
        if 0 <= index < len(self.transactions):
            self.transactions[index]['Status'] = new_status
            return True
        return False
    
    def get_all_account_balances(self):
        balances = {}
        
        for transaction in self.transactions:
            account = transaction['account']
            amount = float(transaction['amount'])
            transaction_type = transaction['type'].lower()
            to_account = transaction['to_account']
            
            # Initialize accounts if they don't exist
            if account not in balances:
                balances[account] = 0
            if to_account != "None" and to_account not in balances:
                balances[to_account] = 0
            
            # Update balances based on transaction type
            if transaction_type == "credit":
                balances[account] += amount
            elif transaction_type == "debit":
                balances[account] -= amount
            elif transaction_type == "transferred":
                balances[account] -= amount
                if to_account != "None":
                    balances[to_account] += amount
        
        # Clean up any NaN values
        cleaned_balances = {}
        for account, balance in balances.items():
            if pd.isna(account) or pd.isna(balance):
                # Skip NaN keys or values
                continue
            cleaned_balances[str(account)] = float(balance) if not pd.isna(balance) else 0.0
        
        return cleaned_balances
    
    def get_spending_by_category(self, start_date=None, end_date=None):
        # Convert to pandas DataFrame for easier filtering and grouping
        df = pd.DataFrame(self.transactions)
        
        if len(df) == 0:
            return {}
        
        # Filter by date if provided
        if start_date and end_date:
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
            start = pd.to_datetime(start_date, format='%d-%m-%Y')
            end = pd.to_datetime(end_date, format='%d-%m-%Y')
            df = df[(df['date'] >= start) & (df['date'] <= end)]
        
        # Filter only debit transactions (expenses)
        debit_df = df[df['type'].str.lower() == 'debit']
        
        if len(debit_df) == 0:
            return {}
        
        # Group by category and sum amounts
        category_spending = debit_df.groupby('category')['amount'].sum().to_dict()
        
        return category_spending
    
    def plot_spending_by_category(self, start_date=None, end_date=None):
        category_spending = self.get_spending_by_category(start_date, end_date)
        
        if not category_spending:
            return None
        
        plt.figure(figsize=(10, 6))
        plt.bar(category_spending.keys(), category_spending.values())
        plt.xlabel('Category')
        plt.ylabel('Amount Spent')
        plt.title('Spending by Category')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot to a bytes buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()
        
        # Convert to base64 for easy transmission
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return img_str
    
    def plot_monthly_spending(self):
        df = pd.DataFrame(self.transactions)
        
        if len(df) == 0:
            return None
        
        # Convert date strings to datetime objects
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
        
        # Filter only debit transactions
        debit_df = df[df['Type'].str.lower() == 'debit']
        
        if len(debit_df) == 0:
            return None
        
        # Extract month and year
        debit_df['Month'] = debit_df['date'].dt.strftime('%Y-%m')
        
        # Group by month and sum
        monthly_spending = debit_df.groupby('Month')['Amount'].sum()
        
        plt.figure(figsize=(12, 6))
        monthly_spending.plot(kind='bar')
        plt.xlabel('Month')
        plt.ylabel('Amount Spent')
        plt.title('Monthly Spending')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot to a bytes buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()
        
        # Convert to base64 for easy transmission
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return img_str
    
    def load_from_csv(self, file_path):
        try:
            df = pd.read_csv(file_path)
            self.transactions = df.to_dict('records')
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def save_to_csv(self, file_path):
        try:
            df = pd.DataFrame(self.transactions)
            df.to_csv(file_path, index=False)
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False


# Initialize tracker
tracker = ExpenseTracker()
file_path = 'expenses.csv'

# Load existing data if file exists
if os.path.exists(file_path):
    tracker.load_from_csv(file_path)


# API Routes

@app.get("/api/transactions", response_model=TransactionListResponse, summary="Get recent transactions")
async def get_transactions(limit: int = Query(10, description="Number of transactions to return")):
    """
    Get recent transactions with optional limit parameter.
    """
    try:
        transactions = tracker.transactions[-limit:] if limit < len(tracker.transactions) else tracker.transactions
        return {"success": True, "transactions": transactions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/transactions", response_model=SuccessResponse, summary="Add a new transaction")
async def add_transaction(transaction: TransactionCreate):
    """
    Add a new transaction with the provided details.
    """
    try:
        index = tracker.add_transaction(
            transaction.date,
            transaction.description,
            transaction.name,
            transaction.amount,
            transaction.type,
            transaction.category,
            transaction.account,
            transaction.to_account,
            transaction.paid_by,
            transaction.status
        )
        
        # Save to CSV
        tracker.save_to_csv(file_path)
        
        return {"success": True, "message": f"Transaction added with index {index}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/transactions/{index}", response_model=SuccessResponse, summary="Delete a transaction")
async def delete_transaction(index: int):
    """
    Delete a transaction by its index.
    """
    try:
        if tracker.delete_transaction(index):
            tracker.save_to_csv(file_path)
            return {"success": True, "message": "Transaction deleted"}
        else:
            raise HTTPException(status_code=404, detail="Transaction not found")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/transactions/{index}/status", response_model=SuccessResponse, summary="Update transaction status")
async def update_status(index: int, update_data: TransactionUpdate):
    """
    Update the status of a transaction.
    """
    try:
        if tracker.update_transaction_status(index, update_data.status):
            tracker.save_to_csv(file_path)
            return {"success": True, "message": "Status updated"}
        else:
            raise HTTPException(status_code=404, detail="Transaction not found")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/balances", response_model=BalanceResponse, summary="Get account balances")
async def get_balances():
    """
    Get balances for all accounts.
    """
    try:
        balances = tracker.get_all_account_balances()
        return {"success": True, "balances": balances}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/spending/category", response_model=SpendingCategoryResponse, summary="Get spending by category")
async def get_spending_by_category(
    start_date: Optional[str] = Query(None, description="Start date in DD-MM-YYYY format"),
    end_date: Optional[str] = Query(None, description="End date in DD-MM-YYYY format")
):
    """
    Get spending breakdown by category with optional date filtering.
    """
    try:
        spending = tracker.get_spending_by_category(start_date, end_date)
        return {"success": True, "spending": spending}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/charts/category", response_model=ChartResponseModel, summary="Get category spending chart")
async def get_category_chart(
    start_date: Optional[str] = Query(None, description="Start date in DD-MM-YYYY format"),
    end_date: Optional[str] = Query(None, description="End date in DD-MM-YYYY format")
):
    """
    Get a chart showing spending by category. Returns base64 encoded PNG image.
    """
    try:
        img_str = tracker.plot_spending_by_category(start_date, end_date)
        if img_str:
            return {"success": True, "chart": img_str}
        else:
            raise HTTPException(status_code=404, detail="No data available for chart")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/charts/monthly", response_model=ChartResponseModel, summary="Get monthly spending chart")
async def get_monthly_chart():
    """
    Get a chart showing monthly spending. Returns base64 encoded PNG image.
    """
    try:
        img_str = tracker.plot_monthly_spending()
        if img_str:
            return {"success": True, "chart": img_str}
        else:
            raise HTTPException(status_code=404, detail="No data available for chart")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)