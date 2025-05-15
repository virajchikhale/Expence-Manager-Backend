from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, EmailStr
from typing import List, Dict, Optional, Union
from passlib.context import CryptContext
from datetime import datetime, timedelta
import jwt
from jwt.exceptions import PyJWTError
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd
import os
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi
from bson import ObjectId
from fastapi.middleware.cors import CORSMiddleware

# Constants and configuration
MONGO_URI = "mongodb+srv://viraj:VirajDB99@finance.kjg2zf7.mongodb.net/?retryWrites=true&w=majority&appName=Finance"
DB_NAME = "expense_tracker_db"
SECRET_KEY = "YOUR_SECRET_KEY"  # Replace with a secure key in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Lifespan context manager for MongoDB connection
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Connect to MongoDB
    app.mongodb_client = AsyncIOMotorClient(MONGO_URI, server_api=ServerApi('1'))
    app.mongodb = app.mongodb_client[DB_NAME]
    
    # Check connection
    try:
        # Send a ping to confirm a successful connection
        await app.mongodb_client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB Atlas!")
        
        # Create a unique index on email field
        await app.mongodb["users"].create_index("email", unique=True)
    except Exception as e:
        print(f"MongoDB connection error: {e}")
        raise
    
    yield  # This is where FastAPI serves requests
    
    # Shutdown: Close MongoDB connection
    app.mongodb_client.close()
    print("MongoDB connection closed.")

# FastAPI app with lifespan
app = FastAPI(title="Expense Tracker API", description="API for tracking personal expenses", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Models
class UserBase(BaseModel):
    email: EmailStr
    username: str
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str

class UserInDB(UserBase):
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class User(UserBase):
    id: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

# Transaction models
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
    id: str

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

class UserListResponse(BaseModel):
    success: bool = True
    users: List[User]

# Helper functions
def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_user_by_email(app, email: str):
    user = await app.mongodb["users"].find_one({"email": email})
    if user:
        return user
    return None

async def authenticate_user(app, email: str, password: str):
    user = await get_user_by_email(app, email)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except PyJWTError:
        raise credentials_exception
    
    user = await get_user_by_email(app, email)
    if user is None:
        raise credentials_exception
    return user

# Expense Tracker class that uses MongoDB
class ExpenseTracker:
    def __init__(self, app, user_id):
        self.app = app
        self.user_id = user_id
        self.transactions_collection = app.mongodb["transactions"]
    
    async def add_transaction(self, date, description, name, amount, transaction_type, 
                       category, account_name, to_account="None", paid_by="Self", status="Pending"):
        transaction = {
            'user_id': self.user_id,
            'date': date,
            'description': description,
            'name': name,
            'amount': amount,
            'type': transaction_type,
            'category': category,
            'account': account_name,
            'to_account': to_account,
            'paid_by': paid_by,
            'status': status,
            'created_at': datetime.utcnow()
        }
        
        result = await self.transactions_collection.insert_one(transaction)
        return str(result.inserted_id)
    
    async def delete_transaction(self, transaction_id):
        try:
            result = await self.transactions_collection.delete_one({
                "_id": ObjectId(transaction_id),
                "user_id": self.user_id
            })
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting transaction: {e}")
            return False
    
    async def update_transaction_status(self, transaction_id, new_status):
        try:
            result = await self.transactions_collection.update_one(
                {"_id": ObjectId(transaction_id), "user_id": self.user_id},
                {"$set": {"status": new_status}}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating transaction: {e}")
            return False
    
    async def get_transactions(self, limit=None):
        cursor = self.transactions_collection.find({"user_id": self.user_id})
        
        # Sort by date descending
        cursor = cursor.sort("created_at", -1)
        
        # Apply limit if provided
        if limit:
            cursor = cursor.limit(limit)
        
        transactions = []
        async for doc in cursor:
            # Convert ObjectId to string
            doc["id"] = str(doc["_id"])
            del doc["_id"]
            transactions.append(doc)
        
        return transactions
    
    async def get_all_account_balances(self):
        transactions = await self.get_transactions()
        balances = {}
        
        for transaction in transactions:
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
        
        return balances
    
    async def get_spending_by_category(self, start_date=None, end_date=None):
        # Get all transactions
        transactions = await self.get_transactions()
        
        if not transactions:
            return {}
        
        # Convert to pandas DataFrame for easier filtering and grouping
        df = pd.DataFrame(transactions)
        
        # Filter by date if provided
        if start_date and end_date:
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
            start = pd.to_datetime(start_date, format='%d-%m-%Y')
            end = pd.to_datetime(end_date, format='%d-%m-%Y')
            df = df[(df['date'] >= start) & (df['date'] <= end)]
        
        # Filter only debit transactions (expenses)
        if not df.empty:
            debit_df = df[df['type'].str.lower() == 'debit']
            
            if not debit_df.empty:
                # Group by category and sum amounts
                category_spending = debit_df.groupby('category')['amount'].sum().to_dict()
                return category_spending
        
        return {}
    
    async def plot_spending_by_category(self, start_date=None, end_date=None):
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
        
        # Save plot to a bytes buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()
        
        # Convert to base64 for easy transmission
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return img_str
    
    async def plot_monthly_spending(self):
        transactions = await self.get_transactions()
        
        if not transactions:
            return None
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(transactions)
        
        # Convert date strings to datetime objects
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
        
        # Filter only debit transactions
        if not df.empty:
            debit_df = df[df['type'].str.lower() == 'debit']
            
            if not debit_df.empty:
                # Extract month and year
                debit_df['Month'] = debit_df['date'].dt.strftime('%Y-%m')
                
                # Group by month and sum
                monthly_spending = debit_df.groupby('Month')['amount'].sum()
                
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
        
        return None

# Get tracker instance for specific user
async def get_tracker(user):
    user_id = str(user["_id"])
    return ExpenseTracker(app, user_id)

# Authentication endpoints
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(app, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"]}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

# User management endpoints
@app.post("/api/users", response_model=User, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate):
    # Check if user already exists
    existing_user = await get_user_by_email(app, user.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user.password)
    user_data = UserInDB(
        email=user.email,
        username=user.username,
        full_name=user.full_name,
        hashed_password=hashed_password
    )
    
    new_user = await app.mongodb["users"].insert_one(user_data.dict())
    
    created_user = await app.mongodb["users"].find_one({"_id": new_user.inserted_id})
    
    return {
        "id": str(created_user["_id"]),
        "email": created_user["email"],
        "username": created_user["username"],
        "full_name": created_user.get("full_name")
    }

@app.get("/api/users/me", response_model=User)
async def read_users_me(current_user = Depends(get_current_user)):
    return {
        "id": str(current_user["_id"]),
        "email": current_user["email"],
        "username": current_user["username"],
        "full_name": current_user.get("full_name")
    }

@app.get("/api/users", response_model=UserListResponse)
async def get_users(current_user: dict = Depends(get_current_user)):
    # In a real app, you'd want to check if current_user has admin privileges
    users = []
    async for user in app.mongodb["users"].find():
        users.append(User(
            id=str(user["_id"]),
            email=user["email"],
            username=user["username"],
            full_name=user.get("full_name")
        ))
    
    return {"success": True, "users": users}

# Transaction endpoints
@app.get("/api/transactions", response_model=TransactionListResponse)
async def get_transactions(
    limit: int = Query(10, description="Number of transactions to return"),
    current_user: dict = Depends(get_current_user)
):
    """Get recent transactions for the authenticated user."""
    try:
        tracker = await get_tracker(current_user)
        transactions = await tracker.get_transactions(limit)
        return {"success": True, "transactions": transactions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/transactions", response_model=SuccessResponse)
async def add_transaction(
    transaction: TransactionCreate,
    current_user: dict = Depends(get_current_user)
):
    """Add a new transaction for the authenticated user."""
    try:
        tracker = await get_tracker(current_user)
        transaction_id = await tracker.add_transaction(
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
        return {"success": True, "message": f"Transaction added with ID {transaction_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/transactions/{transaction_id}", response_model=SuccessResponse)
async def delete_transaction(
    transaction_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a transaction for the authenticated user."""
    try:
        tracker = await get_tracker(current_user)
        if await tracker.delete_transaction(transaction_id):
            return {"success": True, "message": "Transaction deleted"}
        else:
            raise HTTPException(status_code=404, detail="Transaction not found")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/transactions/{transaction_id}/status", response_model=SuccessResponse)
async def update_status(
    transaction_id: str,
    update_data: TransactionUpdate,
    current_user: dict = Depends(get_current_user)
):
    """Update transaction status for the authenticated user."""
    try:
        tracker = await get_tracker(current_user)
        if await tracker.update_transaction_status(transaction_id, update_data.status):
            return {"success": True, "message": "Status updated"}
        else:
            raise HTTPException(status_code=404, detail="Transaction not found")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/balances", response_model=BalanceResponse)
async def get_balances(current_user: dict = Depends(get_current_user)):
    """Get account balances for the authenticated user."""
    try:
        tracker = await get_tracker(current_user)
        balances = await tracker.get_all_account_balances()
        return {"success": True, "balances": balances}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/spending/category", response_model=SpendingCategoryResponse)
async def get_spending_by_category(
    start_date: Optional[str] = Query(None, description="Start date in DD-MM-YYYY format"),
    end_date: Optional[str] = Query(None, description="End date in DD-MM-YYYY format"),
    current_user: dict = Depends(get_current_user)
):
    """Get spending by category for the authenticated user."""
    try:
        tracker = await get_tracker(current_user)
        spending = await tracker.get_spending_by_category(start_date, end_date)
        return {"success": True, "spending": spending}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/charts/category", response_model=ChartResponseModel)
async def get_category_chart(
    start_date: Optional[str] = Query(None, description="Start date in DD-MM-YYYY format"),
    end_date: Optional[str] = Query(None, description="End date in DD-MM-YYYY format"),
    current_user: dict = Depends(get_current_user)
):
    """Get category spending chart for the authenticated user."""
    try:
        tracker = await get_tracker(current_user)
        img_str = await tracker.plot_spending_by_category(start_date, end_date)
        if img_str:
            return {"success": True, "chart": img_str}
        else:
            raise HTTPException(status_code=404, detail="No data available for chart")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/charts/monthly", response_model=ChartResponseModel)
async def get_monthly_chart(current_user: dict = Depends(get_current_user)):
    """Get monthly spending chart for the authenticated user."""
    try:
        tracker = await get_tracker(current_user)
        img_str = await tracker.plot_monthly_spending()
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)