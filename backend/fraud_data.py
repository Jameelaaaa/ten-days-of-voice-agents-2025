#!/usr/bin/env python3
"""
Fraud Cases Database Viewer
A simple utility to view and manage fraud cases data from the SQLite database.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Database path
DB_PATH = Path(__file__).parent / "shared-data" / "fraud_cases.db"

def get_database_connection():
    """Get a connection to the fraud database."""
    return sqlite3.connect(DB_PATH, timeout=30.0)

def view_all_cases():
    """Display all fraud cases in a readable format."""
    try:
        conn = get_database_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, user_name, security_identifier, card_ending, case_status,
                   transaction_name, transaction_time, transaction_amount,
                   transaction_category, transaction_source, location,
                   last_updated, outcome_note
            FROM fraud_cases
            ORDER BY last_updated DESC
        """)
        
        cases = cursor.fetchall()
        conn.close()
        
        if not cases:
            print("📄 No fraud cases found in database.")
            return
        
        print(f"🛡️  FRAUD CASES DATABASE - {len(cases)} cases found")
        print("=" * 80)
        
        for case in cases:
            print(f"""
              🔍 Case ID: {case[0]}
              👤 Customer: {case[1]}
              🏦 Security ID: {case[2]}
              💳 Card Ending: ****{case[3]}
              📊 Status: {case[4].upper()}
              💰 Transaction: {case[7]} at {case[5]}
              🏪 Category: {case[8]} | Source: {case[9]}
              📍 Location: {case[10]}
              ⏰ Last Updated: {case[11]}
              📝 Outcome: {case[12] or 'N/A'}
              {'-' * 40}""")
            
    except Exception as e:
        print(f"❌ Error viewing cases: {e}")

def view_case_by_name(user_name: str):
    """Display a specific fraud case by user name."""
    try:
        conn = get_database_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, user_name, security_identifier, card_ending, case_status,
                   transaction_name, transaction_time, transaction_amount,
                   transaction_category, transaction_source, security_question,
                   security_answer, location, last_updated, outcome_note
            FROM fraud_cases
            WHERE LOWER(user_name) = LOWER(?)
        """, (user_name,))
        
        case = cursor.fetchone()
        conn.close()
        
        if not case:
            print(f"❌ No fraud case found for user: {user_name}")
            return
        
        print(f"🔍 FRAUD CASE DETAILS FOR {case[1].upper()}")
        print("=" * 60)
        print(f"""
          📋 Case ID: {case[0]}
          👤 Customer Name: {case[1]}
          🏦 Security Identifier: {case[2]}
          💳 Card Ending: ****{case[3]}
          📊 Current Status: {case[4].upper()}

          💳 TRANSACTION DETAILS:
          💰 Amount: {case[7]}
          🏪 Merchant: {case[5]}
          📅 Time: {case[6]}
          🏷️  Category: {case[8]}
          🌐 Source: {case[9]}
          📍 Location: {case[12]}

          🔐 SECURITY INFO:
          ❓ Security Question: {case[10]}
          ✅ Security Answer: {case[11]}

          📊 CASE STATUS:
          ⏰ Last Updated: {case[13]}
          📝 Outcome Note: {case[14] or 'No outcome recorded yet'}
          """)
        
    except Exception as e:
        print(f"❌ Error viewing case for {user_name}: {e}")

def view_cases_by_status(status: str):
    """Display all cases with a specific status."""
    try:
        conn = get_database_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT user_name, transaction_amount, transaction_name, 
                   case_status, last_updated
            FROM fraud_cases
            WHERE LOWER(case_status) = LOWER(?)
            ORDER BY last_updated DESC
        """, (status,))
        
        cases = cursor.fetchall()
        conn.close()
        
        if not cases:
            print(f"📄 No cases found with status: {status}")
            return
        
        print(f"📊 CASES WITH STATUS: {status.upper()} ({len(cases)} found)")
        print("=" * 60)
        
        for case in cases:
            print(f"👤 {case[0]} | 💰 {case[1]} | 🏪 {case[2]} | ⏰ {case[4]}")
        
    except Exception as e:
        print(f"❌ Error viewing cases by status: {e}")

def get_database_stats():
    """Display database statistics."""
    try:
        conn = get_database_connection()
        cursor = conn.cursor()
        
        # Total cases
        cursor.execute("SELECT COUNT(*) FROM fraud_cases")
        total_cases = cursor.fetchone()[0]
        
        # Cases by status
        cursor.execute("""
            SELECT case_status, COUNT(*) 
            FROM fraud_cases 
            GROUP BY case_status
        """)
        status_counts = cursor.fetchall()
        
        # High value transactions (>$1000)
        cursor.execute("""
            SELECT COUNT(*) FROM fraud_cases 
            WHERE CAST(REPLACE(REPLACE(transaction_amount, '$', ''), ',', '') AS REAL) > 1000
        """)
        high_value_count = cursor.fetchone()[0]
        
        conn.close()
        
        print("📊 FRAUD DATABASE STATISTICS")
        print("=" * 40)
        print(f"📁 Total Cases: {total_cases}")
        print(f"💰 High Value Cases (>$1000): {high_value_count}")
        print("\n📈 Cases by Status:")
        
        for status, count in status_counts:
            emoji = "🟡" if status == "pending_review" else "🟢" if "safe" in status else "🔴"
            print(f"  {emoji} {status.replace('_', ' ').title()}: {count}")
            
    except Exception as e:
        print(f"❌ Error getting database stats: {e}")

def export_to_json():
    """Export all fraud cases to the existing JSON file, overwriting it."""
    try:
        conn = get_database_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT user_name, security_identifier, card_ending, case_status,
                   transaction_name, transaction_time, transaction_amount,
                   transaction_category, transaction_source, security_question,
                   security_answer, location, last_updated, outcome_note
            FROM fraud_cases
        """)
        
        cases = cursor.fetchall()
        conn.close()
        
        # Convert to JSON format matching the original structure
        json_data = {
            "fraud_cases": []
        }
        
        for case in cases:
            json_data["fraud_cases"].append({
                "userName": case[0],
                "securityIdentifier": case[1],
                "cardEnding": case[2],
                "case": case[3],
                "transactionName": case[4],
                "transactionTime": case[5],
                "transactionAmount": case[6],
                "transactionCategory": case[7],
                "transactionSource": case[8],
                "securityQuestion": case[9],
                "securityAnswer": case[10],
                "location": case[11],
                "lastUpdated": case[12],
                "outcomeNote": case[13]
            })
        
        # Overwrite the existing JSON file
        json_file = Path(__file__).parent / "shared-data" / "fraud_cases.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"✅ Updated fraud_cases.json with {len(cases)} current cases from database")
        print(f"📁 File location: {json_file}")
        
    except Exception as e:
        print(f"❌ Error updating JSON file: {e}")

def main():
    """Main menu for the fraud cases viewer."""
    while True:
        print("\n🛡️  FRAUD CASES DATABASE VIEWER")
        print("=" * 40)
        print("1. 📋 View All Cases")
        print("2. 🔍 View Case by Name")
        print("3. 📊 View Cases by Status")
        print("4. 📈 Database Statistics")
        print("5. 💾 Update fraud_cases.json")
        print("6. ❌ Exit")
        
        choice = input("\nSelect an option (1-6): ").strip()
        
        if choice == "1":
            view_all_cases()
        elif choice == "2":
            name = input("Enter customer name: ").strip()
            if name:
                view_case_by_name(name)
        elif choice == "3":
            print("Available statuses: pending_review, confirmed_safe, confirmed_fraud, verification_failed")
            status = input("Enter status: ").strip()
            if status:
                view_cases_by_status(status)
        elif choice == "4":
            get_database_stats()
        elif choice == "5":
            export_to_json()
        elif choice == "6":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-6.")

if __name__ == "__main__":
    if not DB_PATH.exists():
        print(f"❌ Database not found at: {DB_PATH}")
        print("Run the fraud agent first to initialize the database.")
    else:
        main()