import logging
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from livekit.agents import ( #type: ignore
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation #type: ignore
from livekit.plugins.turn_detector.multilingual import MultilingualModel #type: ignore
from livekit.agents import ChatContext #type: ignore

logger = logging.getLogger("agent")

load_dotenv(".env.local")


def init_fraud_database():
    """Initialize the fraud cases database with the required schema."""
    db_path = CONFIG["fraud_database"]
    
    # Create the directory if it doesn't exist
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create the fraud_cases table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fraud_cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT NOT NULL,
            security_identifier TEXT,
            card_ending TEXT,
            case_status TEXT DEFAULT 'pending_review',
            transaction_name TEXT,
            transaction_time TEXT,
            transaction_amount TEXT,
            transaction_category TEXT,
            transaction_source TEXT,
            security_question TEXT,
            security_answer TEXT,
            location TEXT,
            last_updated TEXT,
            outcome_note TEXT
        )
    """)
    
    # Check if we have any data, if not, load from JSON file
    cursor.execute("SELECT COUNT(*) FROM fraud_cases")
    if cursor.fetchone()[0] == 0:
        # Load data from existing JSON file
        json_file = Path(__file__).parent.parent / "shared-data" / "fraud_cases.json"
        
        try:
            if json_file.exists():
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # Convert JSON data to database format
                cases_to_insert = []
                for case in json_data.get("fraud_cases", []):
                    cases_to_insert.append((
                        case.get("userName"),
                        case.get("securityIdentifier"),
                        case.get("cardEnding"),
                        case.get("case", "pending_review"),
                        case.get("transactionName"),
                        case.get("transactionTime"),
                        case.get("transactionAmount"),
                        case.get("transactionCategory"),
                        case.get("transactionSource"),
                        case.get("securityQuestion"),
                        case.get("securityAnswer"),
                        case.get("location"),
                        case.get("lastUpdated"),
                        case.get("outcomeNote")
                    ))
                
                if cases_to_insert:
                    cursor.executemany("""
                        INSERT INTO fraud_cases (
                            user_name, security_identifier, card_ending, case_status,
                            transaction_name, transaction_time, transaction_amount,
                            transaction_category, transaction_source, security_question,
                            security_answer, location, last_updated, outcome_note
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, cases_to_insert)
                    
                    logger.info(f"Initialized fraud database with {len(cases_to_insert)} cases from fraud_cases.json")
                else:
                    logger.warning("No fraud cases found in JSON file")
            else:
                logger.warning(f"JSON file not found: {json_file}, creating empty database")
                
        except Exception as e:
            logger.error(f"Error loading data from JSON file: {e}")
            logger.info("Database initialized but empty due to JSON loading error")
    
    conn.commit()
    conn.close()


def get_database_connection():
    """Get a connection to the fraud database."""
    return sqlite3.connect(CONFIG["fraud_database"], timeout=30.0)


def log_fraud_cases():
    """Log all fraud cases for debugging/viewing purposes."""
    try:
        conn = get_database_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT user_name, case_status, transaction_amount, transaction_name,
                   location, last_updated, outcome_note
            FROM fraud_cases
            ORDER BY last_updated DESC
        """)
        
        cases = cursor.fetchall()
        conn.close()
        
        logger.info(f"=== FRAUD CASES DATABASE ({len(cases)} cases) ===")
        for case in cases:
            status_emoji = "🟡" if case[1] == "pending_review" else "🟢" if "safe" in case[1] else "🔴"
            outcome_text = f" | Outcome: {case[6]}" if case[6] else ""
            logger.info(f"{status_emoji} {case[0]}: {case[2]} at {case[3]} ({case[4]}) | Status: {case[1]}{outcome_text}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Error logging fraud cases: {e}")


def log_database_stats():
    """Log database statistics."""
    try:
        conn = get_database_connection()
        cursor = conn.cursor()
        
        # Get status counts
        cursor.execute("""
            SELECT case_status, COUNT(*) 
            FROM fraud_cases 
            GROUP BY case_status
        """)
        status_counts = cursor.fetchall()
        
        conn.close()
        
        logger.info("📊 DATABASE STATS:")
        for status, count in status_counts:
            emoji = "🟡" if status == "pending_review" else "🟢" if "safe" in status else "🔴"
            logger.info(f"  {emoji} {status.replace('_', ' ').title()}: {count}")
        
    except Exception as e:
        logger.error(f"Error logging database stats: {e}")


# Configuration constants - can be moved to env vars or config file
CONFIG = {
    "stt_model": "nova-3",
    "llm_model": "gemini-2.5-flash",
    "voice": "en-US-matthew",  # Professional voice for fraud alert
    "fraud_database": Path(__file__).parent.parent / "shared-data" / "fraud_cases.db"
}


@dataclass
class FraudCase:
    """Data structure for fraud case information."""
    userName: Optional[str] = None
    securityIdentifier: Optional[str] = None
    cardEnding: Optional[str] = None
    case: str = "pending_review"  # pending_review, confirmed_safe, confirmed_fraud, verification_failed
    transactionName: Optional[str] = None
    transactionTime: Optional[str] = None
    transactionAmount: Optional[str] = None
    transactionCategory: Optional[str] = None
    transactionSource: Optional[str] = None
    securityQuestion: Optional[str] = None
    securityAnswer: Optional[str] = None
    location: Optional[str] = None
    lastUpdated: str = field(default_factory=lambda: datetime.now().isoformat())
    outcomeNote: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert fraud case to dictionary for JSON serialization."""
        return {
            'userName': self.userName,
            'securityIdentifier': self.securityIdentifier,
            'cardEnding': self.cardEnding,
            'case': self.case,
            'transactionName': self.transactionName,
            'transactionTime': self.transactionTime,
            'transactionAmount': self.transactionAmount,
            'transactionCategory': self.transactionCategory,
            'transactionSource': self.transactionSource,
            'securityQuestion': self.securityQuestion,
            'securityAnswer': self.securityAnswer,
            'location': self.location,
            'lastUpdated': self.lastUpdated,
            'outcomeNote': self.outcomeNote
        }


@dataclass
class FraudData:
    """Shared data for the fraud alert session."""
    current_fraud_case: Optional[FraudCase] = None
    all_fraud_cases: list[Dict[str, Any]] = field(default_factory=list)
    verification_status: str = "pending"  # pending, verified, failed
    call_stage: str = "greeting"  # greeting, verification, transaction_review, case_update, closing


class BaseAgent(Agent):
    """Base agent class with common initialization."""
    
    def __init__(self, instructions: str, chat_ctx: Optional[ChatContext] = None) -> None:
        super().__init__(
            instructions=instructions,
            chat_ctx=chat_ctx,
            stt=deepgram.STT(model=CONFIG["stt_model"]),
            llm=google.LLM(model=CONFIG["llm_model"]),
            tts=murf.TTS(
                voice=CONFIG["voice"],
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
            vad=silero.VAD.load(),
        )


class FraudAlertAgent(BaseAgent):
    """Fraud alert agent for handling suspicious transaction verification."""
    
    def __init__(self, chat_ctx: Optional[ChatContext] = None) -> None:
        instructions = """You are Aura, a professional fraud detection representative from SecureBank's Fraud Prevention Department.

        Your role is to:
        1. Greet the customer and explain you're calling about a suspicious transaction
        2. Verify the customer's identity using ONLY the security question stored in their fraud case
        3. Read out the suspicious transaction details clearly
        4. Ask if they made this transaction
        5. Update the case status based on their response
        6. Provide appropriate next steps

        CRITICAL SECURITY GUIDELINES:
        - NEVER ask for full card numbers, PINs, passwords, or SSNs
        - NEVER generate your own security questions - ALWAYS use the verify_customer function to ask the question from the database
        - NEVER ask random or made-up security questions
        - Only use the security question that is already stored in the customer's fraud case
        - Keep verification simple and non-sensitive
        - Be professional, calm, and reassuring
        - Clearly identify yourself and your bank at the start

        FUNCTION TOOL USAGE:
        - After loading a fraud case with load_fraud_case, ALWAYS call ask_security_question() to ask the stored security question
        - Do NOT generate or ask any security questions yourself
        - Do NOT ask for mother's maiden name, favorite color, or any other questions not in the database
        - The ask_security_question function will provide the exact question to ask
        - After the customer answers, call verify_customer with their answer

        CALL FLOW:
        1. Introduce yourself: "Hello, this is Aurora from SecureBank's Fraud Prevention Department"
        2. Ask for the customer's name to load their case using load_fraud_case
        3. After loading the case, immediately call ask_security_question() to ask the stored security question
        4. Wait for customer response, then call verify_customer with their answer
        5. If verified, call read_transaction_details to show the transaction
        6. Ask if they made the transaction and call update_fraud_case with their response
        7. End the call professionally

        Always maintain a helpful, professional tone and prioritize customer security."""

        super().__init__(instructions=instructions, chat_ctx=chat_ctx)

    @function_tool
    async def load_fraud_case(self, context: RunContext[FraudData], user_name: str) -> str:
        """Load fraud case for the specified user from the database."""
        try:
            conn = get_database_connection()
            cursor = conn.cursor()
            
            # Query for the fraud case
            cursor.execute("""
                SELECT user_name, security_identifier, card_ending, case_status,
                       transaction_name, transaction_time, transaction_amount,
                       transaction_category, transaction_source, security_question,
                       security_answer, location, last_updated, outcome_note
                FROM fraud_cases 
                WHERE LOWER(user_name) = LOWER(?)
            """, (user_name,))
            
            case_data = cursor.fetchone()
            conn.close()
            
            if case_data:
                # Create FraudCase object from database data
                fraud_case = FraudCase(
                    userName=case_data[0],
                    securityIdentifier=case_data[1],
                    cardEnding=case_data[2],
                    case=case_data[3],
                    transactionName=case_data[4],
                    transactionTime=case_data[5],
                    transactionAmount=case_data[6],
                    transactionCategory=case_data[7],
                    transactionSource=case_data[8],
                    securityQuestion=case_data[9],
                    securityAnswer=case_data[10],
                    location=case_data[11],
                    lastUpdated=case_data[12],
                    outcomeNote=case_data[13]
                )
                
                context.userdata.current_fraud_case = fraud_case
                context.userdata.call_stage = "verification"
                
                logger.info(f"Loaded fraud case for {user_name}")
                return f"Thank you {user_name}. I have your account information here. For security purposes, I need to verify your identity before we proceed."
            else:
                return f"I don't see any pending fraud alerts for {user_name}. Could you please double-check the name you provided?"
            
        except Exception as e:
            logger.error(f"Error loading fraud case: {e}")
            return "I'm having trouble accessing our fraud database right now. Please call back in a few minutes."

    @function_tool
    async def ask_security_question(self, context: RunContext[FraudData]) -> str:
        """Ask the customer's security question for verification."""
        fraud_case = context.userdata.current_fraud_case
        
        if not fraud_case:
            return "I need to load your case information first. Could you please tell me your name?"
        
        if context.userdata.verification_status != "pending":
            return "Verification is already in progress."
        
        # Ask the security question
        context.userdata.verification_status = "awaiting_answer"
        return f"For verification, please answer this security question: {fraud_case.securityQuestion}"

    @function_tool
    async def verify_customer(self, context: RunContext[FraudData], customer_answer: str) -> str:
        """Verify customer identity using security question.
        
        Args:
            customer_answer: The customer's answer to the security question.
        """
        fraud_case = context.userdata.current_fraud_case
        
        if not fraud_case:
            return "I need to load your case information first. Could you please tell me your name?"
        
        if context.userdata.verification_status == "verified":
            return "You've already been verified. Let me proceed with the transaction details."
        
        if context.userdata.verification_status != "awaiting_answer":
            return "I need to ask the security question first. Please tell me your name so I can load your case."
        
        # Verify the answer
        if customer_answer.lower().strip() == fraud_case.securityAnswer.lower():
            context.userdata.verification_status = "verified"
            context.userdata.call_stage = "transaction_review"
            return "Thank you for verifying your identity. Now, let me tell you about the suspicious transaction we detected on your account."
        else:
            context.userdata.verification_status = "failed"
            return "I'm sorry, but that doesn't match our records. For your security, I cannot proceed with this call. Please visit your nearest branch with proper identification."

    @function_tool
    async def read_transaction_details(self, context: RunContext[FraudData]) -> str:
        """Read out the suspicious transaction details to the customer."""
        fraud_case = context.userdata.current_fraud_case
        
        if not fraud_case:
            return "I need to verify your identity first."
            
        if context.userdata.verification_status != "verified":
            return "I need to complete identity verification before sharing transaction details."
        
        # Format transaction time for readability
        try:
            from datetime import datetime
            transaction_time = datetime.fromisoformat(fraud_case.transactionTime.replace('Z', '+00:00'))
            formatted_time = transaction_time.strftime("%A, %B %d at %I:%M %p")
        except:
            formatted_time = fraud_case.transactionTime
        
        transaction_details = f"""Here are the details of the suspicious transaction:
        
        Transaction: {fraud_case.transactionAmount} charged to your card ending in {fraud_case.cardEnding}
        Merchant: {fraud_case.transactionName}
        Category: {fraud_case.transactionCategory}
        Location: {fraud_case.location}
        Time: {formatted_time}
        Source: {fraud_case.transactionSource}
        
        Did you make this purchase? Please answer yes if you made this transaction, or no if you did not."""
        
        context.userdata.call_stage = "case_update"
        return transaction_details

    @function_tool
    async def update_fraud_case(self, context: RunContext[FraudData], customer_response: str) -> str:
        """Update the fraud case status based on customer response."""
        fraud_case = context.userdata.current_fraud_case
        
        if not fraud_case or context.userdata.verification_status != "verified":
            return "I need to complete verification first."
        
        response_lower = customer_response.lower().strip()
        
        if "yes" in response_lower or "i made" in response_lower or "i did" in response_lower:
            # Customer confirms transaction is legitimate
            fraud_case.case = "confirmed_safe"
            fraud_case.outcomeNote = "Customer confirmed transaction as legitimate"
            fraud_case.lastUpdated = datetime.now().isoformat()
            
            outcome_message = f"""Perfect! I've marked this transaction as legitimate in our system. 
            
            Your card ending in {fraud_case.cardEnding} remains active and secure. Thank you for helping us keep your account safe. 

            If you have any questions or see any other suspicious activity, please don't hesitate to call us immediately."""
            
        elif "no" in response_lower or "not me" in response_lower or "didn't" in response_lower:
            # Customer denies making the transaction
            fraud_case.case = "confirmed_fraud"
            fraud_case.outcomeNote = "Customer denied making the transaction - fraudulent activity confirmed"
            fraud_case.lastUpdated = datetime.now().isoformat()
            
            outcome_message = f"""I understand. I've immediately flagged this as fraudulent activity and taken the following actions:
            
            1. Your card ending in {fraud_case.cardEnding} has been temporarily blocked to prevent further unauthorized charges
            2. This transaction will be reversed within 3-5 business days
            3. We'll mail you a replacement card within 2-3 business days
            4. A fraud dispute case has been opened

            You should monitor your account closely and report any other suspicious activity immediately. Is there anything else I can help you with regarding this fraud case?"""
            
        else:
            return "I need a clear yes or no answer. Did you make this transaction for $" + str(fraud_case.transactionAmount) + "?"
        
        # Save the updated case back to the database
        try:
            self._save_fraud_case_to_database(context, fraud_case)
            logger.info(f"Updated fraud case for {fraud_case.userName} - Status: {fraud_case.case}")
        except Exception as e:
            logger.error(f"Error saving fraud case: {e}")
        
        context.userdata.call_stage = "closing"
        return outcome_message

    def _save_fraud_case_to_database(self, context: RunContext[FraudData], updated_case: FraudCase):
        """Save the updated fraud case back to the database."""
        try:
            conn = get_database_connection()
            cursor = conn.cursor()
            
            # Update the fraud case in the database
            cursor.execute("""
                UPDATE fraud_cases 
                SET case_status = ?, 
                    last_updated = ?,
                    outcome_note = ?
                WHERE LOWER(user_name) = LOWER(?)
            """, (
                updated_case.case,
                updated_case.lastUpdated,
                updated_case.outcomeNote,
                updated_case.userName
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Successfully updated fraud case for {updated_case.userName} in database")
                
        except Exception as e:
            logger.error(f"Error saving fraud case to database: {e}")
            raise

    @function_tool
    async def end_fraud_call(self, context: RunContext[FraudData]) -> str:
        """End the fraud alert call with appropriate closing message."""
        fraud_case = context.userdata.current_fraud_case
        
        if fraud_case and fraud_case.case in ["confirmed_safe", "confirmed_fraud"]:
            closing_message = f"""Thank you for your time today, {fraud_case.userName}. Your case has been updated and all necessary actions have been taken.
            
            Remember to:
            - Keep your account information secure
            - Never share your PIN or passwords
            - Contact us immediately if you notice any suspicious activity
            - Monitor your statements regularly

            Have a great day and thank you for banking with SecureBank!"""
        else:
            closing_message = "Thank you for calling SecureBank's Fraud Prevention Department. Stay safe!"
        
        return closing_message


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Initialize the fraud database
    init_fraud_database()
    
    # Log database contents for debugging
    log_database_stats()
    log_fraud_cases()
    
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Initialize Fraud data
    userdata = FraudData()
    
    # Create Fraud Alert agent
    fraud_agent = FraudAlertAgent()
    
    # Create the session with the Fraud Alert agent
    session = AgentSession[FraudData](
        userdata=userdata,
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        preemptive_generation=True,
    )

    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session with the fraud agent
    await session.start(
        agent=fraud_agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
