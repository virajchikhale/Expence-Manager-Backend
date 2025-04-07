import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np

class ExpenseTracker:
    def __init__(self, file_path=None):
        self.columns = [
            'Date', 'Description', 'Name', 'Amount', 'Transaction Type',
            'Category', 'Account Name', 'To Account', 'Paid By', 'Status'
        ]
        
        # Initialize DataFrame
        if file_path and os.path.exists(file_path):
            self.df = pd.read_csv(file_path)
        else:
            self.df = pd.DataFrame(columns=self.columns)
        
        # Convert Date column to datetime if it exists and has data
        if 'Date' in self.df.columns and not self.df.empty:
            self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d-%m-%Y', errors='coerce')
    
    def add_transaction(self, date, description, name, amount, transaction_type, category, account_name, to_account="None", paid_by="Self", status="Pending"):
        """Add a new transaction to the record"""
        try:
            # Convert date string to datetime
            if isinstance(date, str):
                date = datetime.strptime(date, '%d-%m-%Y')
            
            new_transaction = pd.DataFrame({
                'Date': [date],
                'Description': [description],
                'Name': [name],
                'Amount': [float(amount)],
                'Transaction Type': [transaction_type],
                'Category': [category],
                'Account Name': [account_name],
                'To Account': [to_account],
                'Paid By': [paid_by],
                'Status': [status]
            })
            
            self.df = pd.concat([self.df, new_transaction], ignore_index=True)
            print(f"Transaction added: {description} for {amount}")
            return True
        except Exception as e:
            print(f"Error adding transaction: {str(e)}")
            return False

    def update_transaction_status(self, index, new_status):
        """Update the status of a transaction"""
        try:
            if 0 <= index < len(self.df):
                self.df.at[index, 'Status'] = new_status
                print(f"Transaction status updated to {new_status}")
                return True
            else:
                print("Invalid transaction index")
                return False
        except Exception as e:
            print(f"Error updating transaction: {str(e)}")
            return False

    def delete_transaction(self, index):
        """Delete a transaction by index"""
        try:
            if 0 <= index < len(self.df):
                description = self.df.iloc[index]['Description']
                self.df = self.df.drop(index).reset_index(drop=True)
                print(f"Transaction deleted: {description}")
                return True
            else:
                print("Invalid transaction index")
                return False
        except Exception as e:
            print(f"Error deleting transaction: {str(e)}")
            return False

    def get_account_balance(self, account_name):
        """Calculate the current balance for a specific account"""
        account_transactions = self.df[self.df['Account Name'] == account_name]
        
        # Filter for transactions that affect the balance
        credits = account_transactions[account_transactions['Transaction Type'].isin(['Credit', 'Received'])]['Amount'].sum()
        debits = account_transactions[account_transactions['Transaction Type'].isin(['Debit', 'Paid'])]['Amount'].sum()
        
        # Handle transfers
        transfers_out = account_transactions[
            (account_transactions['Transaction Type'] == 'Transfered') & 
            (account_transactions['Account Name'] == account_name)
        ]['Amount'].sum()
        
        transfers_in = self.df[
            (self.df['Transaction Type'] == 'Transfered') & 
            (self.df['To Account'] == account_name)
        ]['Amount'].sum()
        
        balance = credits - debits - transfers_out + transfers_in
        return balance

    def get_all_account_balances(self):
        """Get balances for all accounts"""
        accounts = self.df['Account Name'].unique()
        balances = {}
        
        for account in accounts:
            balances[account] = self.get_account_balance(account)
        
        return balances

    def get_spending_by_category(self, start_date=None, end_date=None):
        """Calculate spending by category within date range"""
        df_filtered = self.df.copy()
        
        if start_date:
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%d-%m-%Y')
            df_filtered = df_filtered[df_filtered['Date'] >= start_date]
        
        if end_date:
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%d-%m-%Y')
            df_filtered = df_filtered[df_filtered['Date'] <= end_date]
        
        # Only consider debit transactions for spending
        spending_df = df_filtered[df_filtered['Transaction Type'].isin(['Debit', 'Paid'])]
        
        return spending_df.groupby('Category')['Amount'].sum().sort_values(ascending=False)

    def get_transactions_by_date_range(self, start_date=None, end_date=None):
        """Get transactions within a specific date range"""
        df_filtered = self.df.copy()
        
        if start_date:
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%d-%m-%Y')
            df_filtered = df_filtered[df_filtered['Date'] >= start_date]
        
        if end_date:
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%d-%m-%Y')
            df_filtered = df_filtered[df_filtered['Date'] <= end_date]
        
        return df_filtered

    def save_to_csv(self, file_path):
        """Save the transaction data to a CSV file"""
        try:
            # Convert datetime to string format before saving
            df_to_save = self.df.copy()
            if 'Date' in df_to_save.columns and not df_to_save.empty:
                df_to_save['Date'] = df_to_save['Date'].dt.strftime('%d-%m-%Y')
            
            df_to_save.to_csv(file_path, index=False)
            print(f"Data saved to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving data: {str(e)}")
            return False

    def load_from_csv(self, file_path):
        """Load transaction data from a CSV file"""
        try:
            if os.path.exists(file_path):
                self.df = pd.read_csv(file_path)
                
                # Convert Date column to datetime
                if 'Date' in self.df.columns and not self.df.empty:
                    self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d-%m-%Y', errors='coerce')
                
                print(f"Data loaded from {file_path}")
                return True
            else:
                print(f"File not found: {file_path}")
                return False
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False

    def display_transactions(self, n=10):
        """Display the most recent transactions"""
        if self.df.empty:
            print("No transactions found.")
            return
        
        # Sort by date (newest first) and display the most recent n transactions
        sorted_df = self.df.sort_values(by='Date', ascending=False).head(n)
        
        # Format for display
        display_df = sorted_df.copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%d-%m-%Y')
        
        print("\nRecent Transactions:")
        for idx, row in display_df.iterrows():
            print(f"{idx}. [{row['Date']}] {row['Description']} - {row['Amount']} ({row['Transaction Type']}) - {row['Account Name']} - {row['Status']}")

    def plot_spending_by_category(self, start_date=None, end_date=None):
        """Plot spending by category in a pie chart"""
        spending = self.get_spending_by_category(start_date, end_date)
        
        if spending.empty:
            print("No spending data available for the selected period.")
            return
        
        plt.figure(figsize=(10, 7))
        plt.pie(spending.values, labels=spending.index, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')
        
        date_range = ""
        if start_date and end_date:
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%d-%m-%Y').strftime('%d-%m-%Y')
            else:
                start_date = start_date.strftime('%d-%m-%Y')
                
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%d-%m-%Y').strftime('%d-%m-%Y')
            else:
                end_date = end_date.strftime('%d-%m-%Y')
                
            date_range = f" ({start_date} to {end_date})"
        
        plt.title(f'Spending by Category{date_range}')
        plt.show()

    def plot_monthly_spending(self):
        """Plot monthly spending trend"""
        if self.df.empty:
            print("No transaction data available.")
            return
            
        # Create a month column
        df_with_month = self.df.copy()
        df_with_month['Month'] = df_with_month['Date'].dt.to_period('M')
        
        # Filter for debit transactions
        spending_df = df_with_month[df_with_month['Transaction Type'].isin(['Debit', 'Paid'])]
        
        # Group by month
        monthly_spending = spending_df.groupby('Month')['Amount'].sum()
        
        # Convert period index to datetime for proper plotting
        months = [pd.to_datetime(str(period)) for period in monthly_spending.index]
        
        plt.figure(figsize=(12, 6))
        plt.bar(months, monthly_spending.values)
        plt.xlabel('Month')
        plt.ylabel('Amount')
        plt.title('Monthly Spending')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def main():
    print("\n===== Personal Expense Tracker =====\n")
    
    # Initialize tracker
    tracker = ExpenseTracker()
    file_path = 'expenses.csv'
    
    # Try to load existing data
    if os.path.exists(file_path):
        tracker.load_from_csv(file_path)
    
    
    while True:
        print("\n--- Menu ---")
        print("1. Add new transaction")
        print("2. View recent transactions")
        print("3. View account balances")
        print("4. View spending by category")
        print("5. Update transaction status")
        print("6. Delete transaction")
        print("7. Plot spending by category")
        print("8. Plot monthly spending")
        print("9. Save and exit")
        
        choice = input("\nEnter your choice (1-9): ")
        
        if choice == '1':
            date = input("Date (DD-MM-YYYY): ")
            description = input("Description: ")
            name = input("Name: ")
            amount = float(input("Amount: "))
            
            print("\nTransaction Types: Credit, Debit, Transfered")
            transaction_type = input("Transaction Type: ")
            
            category = input("Category: ")
            account_name = input("Account Name: ")
            
            to_account = "None"
            if transaction_type.lower() == "transfered":
                to_account = input("To Account: ")
                
            paid_by = input("Paid By (default: Self): ") or "Self"
            status = input("Status (default: Pending): ") or "Pending"
            
            tracker.add_transaction(date, description, name, amount, transaction_type, category, account_name, to_account, paid_by, status)
        
        elif choice == '2':
            n = int(input("How many recent transactions to view? (default: 10): ") or 10)
            tracker.display_transactions(n)
        
        elif choice == '3':
            balances = tracker.get_all_account_balances()
            print("\nAccount Balances:")
            for account, balance in balances.items():
                print(f"{account}: {balance}")
        
        elif choice == '4':
            start_date = input("Start Date (DD-MM-YYYY, optional): ") or None
            end_date = input("End Date (DD-MM-YYYY, optional): ") or None
            spending = tracker.get_spending_by_category(start_date, end_date)
            print("\nSpending by Category:")
            print(spending)
        
        elif choice == '5':
            index = int(input("Enter the transaction index to update: "))
            new_status = input("Enter the new status: ")
            tracker.update_transaction_status(index, new_status)
        
        elif choice == '6':
            index = int(input("Enter the transaction index to delete: "))
            tracker.delete_transaction(index)
        
        elif choice == '7':
            start_date = input("Start Date (DD-MM-YYYY, optional): ") or None
            end_date = input("End Date (DD-MM-YYYY, optional): ") or None
            tracker.plot_spending_by_category(start_date, end_date)
        
        elif choice == '8':
            tracker.plot_monthly_spending()
        
        elif choice == '9':
            # Save the data to a CSV file before exiting
            tracker.save_to_csv(file_path)
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

main()
