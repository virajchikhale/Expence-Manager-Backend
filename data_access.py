from typing import Optional, List, Dict
from datetime import date, datetime
from pydantic import BaseModel
from abc import ABC, abstractmethod
import pandas as pd
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi
from bson import ObjectId
import certifi
import sqlite3
import aiosqlite
import os
from contextlib import asynccontextmanager

# Interface for all database implementations
class DatabaseInterface(ABC):
    @abstractmethod
    async def create_user(self, user_data: dict) -> str:
        pass
    
    @abstractmethod
    async def get_user_by_email(self, email: str) -> Optional[dict]:
        pass
    
    @abstractmethod
    async def create_account(self, user_id: str, account_data: dict) -> str:
        pass
    
    @abstractmethod
    async def get_accounts(self, user_id: str) -> List[dict]:
        pass
    
    @abstractmethod
    async def add_transaction(self, user_id: str, transaction_data: dict) -> str:
        pass
    
    @abstractmethod
    async def get_transactions(self, user_id: str, filters: dict = None) -> List[dict]:
        pass
    
    @abstractmethod
    async def delete_transaction(self, user_id: str, transaction_id: str) -> bool:
        pass
    
    @abstractmethod
    async def update_transaction_status(self, user_id: str, transaction_id: str, new_status: str) -> bool:
        pass
    
    @abstractmethod
    async def get_account_balances(self, user_id: str) -> Dict[str, float]:
        pass

# MongoDB implementation
class MongoDB(DatabaseInterface):
    def __init__(self, connection_string: str, db_name: str):
        self.client = AsyncIOMotorClient(connection_string, server_api=ServerApi('1'), tlsCAFile=certifi.where())
        self.db = self.client[db_name]
    
    async def initialize(self):
        try:
            await self.client.admin.command('ping')
            print("Successfully connected to MongoDB!")
            await self.db["users"].create_index("email", unique=True)
            await self.db["accounts"].create_index([("user_id", 1), ("name", 1)], unique=True)
        except Exception as e:
            print(f"MongoDB connection error: {e}")
            raise
    
    async def close(self):
        self.client.close()
    
    async def create_user(self, user_data: dict) -> str:
        result = await self.db["users"].insert_one(user_data)
        return str(result.inserted_id)
    
    async def get_user_by_email(self, email: str) -> Optional[dict]:
        user = await self.db["users"].find_one({"email": email})
        if user:
            user["id"] = str(user["_id"])
            del user["_id"]
        return user
    
    async def create_account(self, user_id: str, account_data: dict) -> str:
        account_data["user_id"] = user_id
        account_data["created_at"] = datetime.utcnow()
        try:
            result = await self.db["accounts"].insert_one(account_data)
            return str(result.inserted_id)
        except Exception as e:
            if "duplicate key" in str(e):
                raise ValueError(f"Account with name '{account_data['name']}' already exists.")
            raise
    
    async def get_accounts(self, user_id: str) -> List[dict]:
        accounts = []
        async for acc in self.db["accounts"].find({"user_id": user_id}):
            acc["id"] = str(acc["_id"])
            del acc["_id"]
            accounts.append(acc)
        return accounts
    
    async def add_transaction(self, user_id: str, transaction_data: dict) -> str:
        transaction_data["user_id"] = user_id
        transaction_data["created_at"] = datetime.utcnow()
        
        # Calculate transaction balance
        prev_transactions = await self.get_transactions(user_id, {"account": transaction_data["account"]})
        prev_balance = 0.0
        for t in prev_transactions:
            t_type = t.get("type", "").lower()
            t_amount = float(t.get("amount", 0))
            t_to_account = t.get("to_account", "None")
            
            if t_type == "credit":
                prev_balance += t_amount
            elif t_type == "debit":
                prev_balance -= t_amount
            elif t_type == "debt_incurred":
                prev_balance -= t_amount
            elif t_type in ["transferred", "self_transferred"]:
                prev_balance -= t_amount
                if t_to_account == transaction_data["account"]:
                    prev_balance += t_amount
        
        # Apply current transaction
        t_type = transaction_data["type"].lower()
        amount = float(transaction_data["amount"])
        to_account = transaction_data.get("to_account", "None")
        
        if t_type == "credit":
            transaction_data["transaction_balance"] = prev_balance + amount
        elif t_type == "debit":
            transaction_data["transaction_balance"] = prev_balance - amount
        elif t_type == "debt_incurred":
            transaction_data["transaction_balance"] = prev_balance - amount
        elif t_type in ["transferred", "self_transferred"]:
            transaction_data["transaction_balance"] = prev_balance - amount
            if to_account == transaction_data["account"]:
                transaction_data["transaction_balance"] += amount
        
        result = await self.db["transactions"].insert_one(transaction_data)
        return str(result.inserted_id)
    
    async def get_transactions(self, user_id: str, filters: dict = None) -> List[dict]:
        query = {"user_id": user_id}
        if filters:
            query.update(filters)
        
        transactions = []
        async for doc in self.db["transactions"].find(query).sort("created_at", -1):
            doc["id"] = str(doc["_id"])
            if isinstance(doc["date"], datetime):
                doc["date"] = doc["date"].strftime("%Y-%m-%d")
            del doc["_id"]
            transactions.append(doc)
        return transactions
    
    async def delete_transaction(self, user_id: str, transaction_id: str) -> bool:
        result = await self.db["transactions"].delete_one({
            "_id": ObjectId(transaction_id),
            "user_id": user_id
        })
        return result.deleted_count > 0
    
    async def update_transaction_status(self, user_id: str, transaction_id: str, new_status: str) -> bool:
        result = await self.db["transactions"].update_one(
            {"_id": ObjectId(transaction_id), "user_id": user_id},
            {"$set": {"status": new_status}}
        )
        return result.modified_count > 0
    
    async def get_account_balances(self, user_id: str) -> Dict[str, float]:
        accounts = await self.get_accounts(user_id)
        balances = {account['name']: 0.0 for account in accounts}
        transactions = await self.get_transactions(user_id)
        
        for transaction in transactions:
            account = transaction['account']
            amount = float(transaction['amount'])
            transaction_type = transaction['type'].lower()
            to_account = transaction['to_account']
            
            if transaction_type == "credit":
                balances[account] += amount
            elif transaction_type == "debit":
                balances[account] -= amount
            elif transaction_type == "debt_incurred":
                balances[account] -= amount
            elif transaction_type in ["transferred", "self_transferred"]:
                balances[account] -= amount
                if to_account != "None":
                    balances[to_account] += amount
        return balances


class SQLiteDatabase(DatabaseInterface):
    def __init__(self, db_path: str = "expense_tracker.db"):
        self.db_path = db_path
    
    async def initialize(self):
        async with aiosqlite.connect(self.db_path) as db:
            # Create tables if they don't exist
            await db.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    username TEXT NOT NULL,
                    full_name TEXT,
                    hashed_password TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS accounts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, name),
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    date TIMESTAMP NOT NULL,
                    description TEXT NOT NULL,
                    place TEXT NOT NULL,
                    amount REAL NOT NULL,
                    type TEXT NOT NULL,
                    category TEXT NOT NULL,
                    account TEXT NOT NULL,
                    to_account TEXT DEFAULT 'None',
                    paid_by TEXT DEFAULT 'Self',
                    status TEXT DEFAULT 'Pending',
                    transaction_balance REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)
            
            await db.commit()
    
    async def close(self):
        pass  # Connection is closed after each operation with aiosqlite
    
    async def _execute(self, query: str, params: tuple = ()):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(query, params)
            await db.commit()
    
    async def _fetch_one(self, query: str, params: tuple = ()):
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None
    
    async def _fetch_all(self, query: str, params: tuple = ()):
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
    
    async def create_user(self, user_data: dict) -> str:
        query = """
            INSERT INTO users (email, username, full_name, hashed_password)
            VALUES (?, ?, ?, ?)
        """
        params = (
            user_data["email"],
            user_data["username"],
            user_data.get("full_name"),
            user_data["hashed_password"]
        )
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(query, params)
            await db.commit()
            return str(cursor.lastrowid)
    
    async def get_user_by_email(self, email: str) -> Optional[dict]:
        query = "SELECT * FROM users WHERE email = ?"
        return await self._fetch_one(query, (email,))
    
    async def create_account(self, user_id: str, account_data: dict) -> str:
        query = """
            INSERT INTO accounts (user_id, name, type)
            VALUES (?, ?, ?)
        """
        params = (
            int(user_id),
            account_data["name"],
            account_data["type"]
        )
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(query, params)
                await db.commit()
                return str(cursor.lastrowid)
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                raise ValueError(f"Account with name '{account_data['name']}' already exists.")
            raise
    
    async def get_accounts(self, user_id: str) -> List[dict]:
        query = "SELECT * FROM accounts WHERE user_id = ?"
        return await self._fetch_all(query, (int(user_id),))
    
    async def add_transaction(self, user_id: str, transaction_data: dict) -> str:
        # Calculate transaction balance
        prev_balance = 0.0
        prev_transactions = await self.get_transactions(user_id, {"account": transaction_data["account"]})
        
        for t in prev_transactions:
            t_type = t.get("type", "").lower()
            t_amount = float(t.get("amount", 0))
            t_to_account = t.get("to_account", "None")
            
            if t_type == "credit":
                prev_balance += t_amount
            elif t_type == "debit":
                prev_balance -= t_amount
            elif t_type == "debt_incurred":
                prev_balance -= t_amount
            elif t_type in ["transferred", "self_transferred"]:
                prev_balance -= t_amount
                if t_to_account == transaction_data["account"]:
                    prev_balance += t_amount
        
        # Apply current transaction
        t_type = transaction_data["type"].lower()
        amount = float(transaction_data["amount"])
        to_account = transaction_data.get("to_account", "None")
        
        if t_type == "credit":
            transaction_balance = prev_balance + amount
        elif t_type == "debit":
            transaction_balance = prev_balance - amount
        elif t_type == "debt_incurred":
            transaction_balance = prev_balance - amount
        elif t_type in ["transferred", "self_transferred"]:
            transaction_balance = prev_balance - amount
            if to_account == transaction_data["account"]:
                transaction_balance += amount
        
        query = """
            INSERT INTO transactions (
                user_id, date, description, place, amount, type, 
                category, account, to_account, paid_by, status, transaction_balance
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            int(user_id),
            transaction_data["date"],
            transaction_data["description"],
            transaction_data["place"],
            transaction_data["amount"],
            transaction_data["type"],
            transaction_data["category"],
            transaction_data["account"],
            transaction_data.get("to_account", "None"),
            transaction_data.get("paid_by", "Self"),
            transaction_data.get("status", "Pending"),
            transaction_balance
        )
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(query, params)
            await db.commit()
            return str(cursor.lastrowid)
    
    async def get_transactions(self, user_id: str, filters: dict = None) -> List[dict]:
        query = "SELECT * FROM transactions WHERE user_id = ?"
        params = [int(user_id)]
        
        if filters:
            conditions = []
            if filters.get("account"):
                conditions.append("account = ?")
                params.append(filters["account"])
            
            if conditions:
                query += " AND " + " AND ".join(conditions)
        
        query += " ORDER BY created_at DESC"
        
        transactions = await self._fetch_all(query, tuple(params))
        
        # Convert datetime to string for consistency
        for t in transactions:
            if isinstance(t["date"], str):
                t["date"] = datetime.fromisoformat(t["date"]).strftime("%Y-%m-%d")
            elif isinstance(t["date"], datetime):
                t["date"] = t["date"].strftime("%Y-%m-%d")
        
        return transactions
    
    async def delete_transaction(self, user_id: str, transaction_id: str) -> bool:
        query = "DELETE FROM transactions WHERE id = ? AND user_id = ?"
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(query, (int(transaction_id), int(user_id)))
            await db.commit()
            return True
    
    async def update_transaction_status(self, user_id: str, transaction_id: str, new_status: str) -> bool:
        query = "UPDATE transactions SET status = ? WHERE id = ? AND user_id = ?"
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(query, (new_status, int(transaction_id), int(user_id)))
            await db.commit()
            return True
    
    async def get_account_balances(self, user_id: str) -> Dict[str, float]:
        accounts = await self.get_accounts(user_id)
        balances = {account['name']: 0.0 for account in accounts}
        transactions = await self.get_transactions(user_id)
        
        for transaction in transactions:
            account = transaction['account']
            amount = float(transaction['amount'])
            transaction_type = transaction['type'].lower()
            to_account = transaction['to_account']
            
            if transaction_type == "credit":
                balances[account] += amount
            elif transaction_type == "debit":
                balances[account] -= amount
            elif transaction_type == "debt_incurred":
                balances[account] -= amount
            elif transaction_type in ["transferred", "self_transferred"]:
                balances[account] -= amount
                if to_account != "None":
                    balances[to_account] += amount
        
        return balances

# Updated factory to support SQLite
class DatabaseFactory:
    @staticmethod
    async def create(db_type: str, connection_string: str = None, db_name: str = None) -> DatabaseInterface:
        if db_type == "mongodb":
            return MongoDB(connection_string, db_name)
        elif db_type == "sqlite":
            db = SQLiteDatabase(connection_string or "expense_tracker.db")
            await db.initialize()
            return db
        else:
            raise ValueError(f"Unsupported database type: {db_type}")