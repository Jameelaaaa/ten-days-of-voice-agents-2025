import json
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

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

# Configuration constants - can be moved to env vars or config file
CONFIG = {
    "stt_model": "nova-3",
    "llm_model": "gemini-2.5-flash",
    "voice": "en-US-matthew",  # Professional male voice for shopping assistant
    "catalog_file": Path(__file__).parent.parent / "shared-data" / "ecommerce_catalog.json",
    "orders_directory": Path(__file__).parent.parent / "orders"
}


@dataclass
class OrderItem:
    """Data structure for individual order items."""
    product_id: str
    quantity: int
    price: float
    name: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order item to dictionary for JSON serialization."""
        return {
            'product_id': self.product_id,
            'quantity': self.quantity,
            'price': self.price,
            'name': self.name,
            'total': self.price * self.quantity
        }

@dataclass
class Order:
    """Data structure for orders."""
    id: str = field(default_factory=lambda: f"order_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    items: List[OrderItem] = field(default_factory=list)
    total: float = 0.0
    currency: str = "INR"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "pending"
    
    def add_item(self, product: Dict[str, Any], quantity: int = 1):
        """Add an item to the order."""
        item = OrderItem(
            product_id=product['id'],
            quantity=quantity,
            price=product['price'],
            name=product['name']
        )
        self.items.append(item)
        self.total += item.price * quantity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'items': [item.to_dict() for item in self.items],
            'total': self.total,
            'currency': self.currency,
            'created_at': self.created_at,
            'status': self.status
        }
    
    def get_summary(self) -> str:
        """Get a summary of the order."""
        if not self.items:
            return "Empty order"
        
        items_summary = []
        for item in self.items:
            items_summary.append(f"{item.quantity}x {item.name}")
        
        return f"Order {self.id}: {', '.join(items_summary)} - Total: ₹{self.total:.2f}"


@dataclass
class EcommerceData:
    """Shared data for the e-commerce session."""
    catalog: Dict[str, Any] = field(default_factory=dict)
    current_order: Optional[Order] = None
    conversation_stage: str = "browsing"  # browsing, selecting, ordering, checkout
    last_search_results: List[Dict[str, Any]] = field(default_factory=list)
    orders_history: List[Order] = field(default_factory=list)


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


class EcommerceAgent(BaseAgent):
    """E-commerce shopping assistant agent following Agentic Commerce Protocol patterns."""
    
    def __init__(self, chat_ctx: Optional[ChatContext] = None) -> None:
        instructions = """You are Aura, a friendly and helpful shopping assistant at AuraMart
         E-Commerce.

        Your role is to:
        1. Help customers browse and discover products in our catalog
        2. Answer questions about product details, pricing, and availability
        3. Assist customers in adding items to their cart and placing orders
        4. Provide order summaries and confirmations
        5. Handle product searches based on categories, price ranges, and attributes

        Available product categories: drinkware, clothing, stationery, bags, accessories

        CRITICAL: Always call the appropriate function when the user:
        - Asks to see products: call browse_catalog() or search_products()
        - Wants to buy something: call add_to_cart()
        - Asks about their order: call view_current_order()
        - Wants to checkout: call place_order()
        - Asks about past orders: call view_order_history()

        Shopping Flow:
        1. Browsing: Help them find products they're interested in
        2. Selection: Show product details and handle questions
        3. Ordering: Add items to cart and manage quantities
        4. Checkout: Complete the order and provide confirmation

        Always be helpful, enthusiastic about our products, and make shopping easy and enjoyable!
        Use the product catalog to provide accurate information about availability, pricing, and features.
        
        When describing products, mention key details like price, material, color, and availability."""

        super().__init__(instructions=instructions, chat_ctx=chat_ctx)

    @function_tool
    async def browse_catalog(self, context: RunContext[EcommerceData], category: Optional[str] = None) -> str:
        """Browse the product catalog, optionally filtered by category."""
        products = context.userdata.catalog.get('products', [])
        
        if category:
            filtered_products = [p for p in products if p.get('category', '').lower() == category.lower()]
            if not filtered_products:
                return f"I couldn't find any products in the '{category}' category. Available categories are: drinkware, clothing, stationery, bags, accessories."
            products_to_show = filtered_products
        else:
            products_to_show = products
        
        # Limit to first 5 products to avoid overwhelming
        products_to_show = products_to_show[:5]
        context.userdata.last_search_results = products_to_show
        
        if not products_to_show:
            return "I'm sorry, but our catalog appears to be empty at the moment."
        
        result = f"Here are some great products{' in ' + category if category else ''} for you:\n\n"
        
        for i, product in enumerate(products_to_show, 1):
            attributes = product.get('attributes', {})
            attr_str = ", ".join([f"{k}: {v}" for k, v in attributes.items()])
            result += f"{i}. **{product['name']}** - ₹{product['price']} ({product.get('stock', 0)} in stock)\n"
            result += f"   {product['description']}\n"
            if attr_str:
                result += f"   Features: {attr_str}\n"
            result += "\n"
        
        result += "Would you like to see more details about any of these products, or would you like me to help you add something to your cart?"
        return result

    @function_tool
    async def search_products(self, context: RunContext[EcommerceData], query: str, max_price: Optional[float] = None) -> str:
        """Search for products based on query terms and optional price filter."""
        products = context.userdata.catalog.get('products', [])
        query_lower = query.lower()
        
        # Search in product name, description, category, and attributes
        matching_products = []
        for product in products:
            # Check if query matches name, description, or category
            if (query_lower in product['name'].lower() or 
                query_lower in product['description'].lower() or
                query_lower in product['category'].lower()):
                matching_products.append(product)
                continue
                
            # Check attributes
            attributes = product.get('attributes', {})
            for attr_value in attributes.values():
                if isinstance(attr_value, str) and query_lower in str(attr_value).lower():
                    matching_products.append(product)
                    break
        
        # Apply price filter if specified
        if max_price is not None:
            matching_products = [p for p in matching_products if p['price'] <= max_price]
        
        if not matching_products:
            return f"I couldn't find any products matching '{query}'{' under ₹' + str(max_price) if max_price else ''}. Try browsing our categories: drinkware, clothing, stationery, bags, accessories."
        
        # Limit results and store for reference
        matching_products = matching_products[:5]
        context.userdata.last_search_results = matching_products
        
        result = f"Found {len(matching_products)} product{'s' if len(matching_products) != 1 else ''} for '{query}':\n\n"
        
        for i, product in enumerate(matching_products, 1):
            attributes = product.get('attributes', {})
            color = attributes.get('color', '')
            size = attributes.get('size', '')
            result += f"{i}. **{product['name']}** - ₹{product['price']} ({product.get('stock', 0)} in stock)\n"
            result += f"   {product['description']}\n"
            if color or size:
                details = []
                if color: details.append(f"Color: {color}")
                if size: details.append(f"Size: {size}")
                result += f"   {', '.join(details)}\n"
            result += "\n"
        
        result += "Would you like to add any of these to your cart? Just let me know which one!"
        return result

    @function_tool
    async def add_to_cart(self, context: RunContext[EcommerceData], product_identifier: str, quantity: int = 1) -> str:
        """Add a product to the shopping cart."""
        # Find the product
        product = None
        products = context.userdata.catalog.get('products', [])
        
        # Try to find by ID first
        for p in products:
            if p['id'] == product_identifier:
                product = p
                break
        
        # If not found by ID, try to find in last search results by index
        if not product and context.userdata.last_search_results:
            try:
                # Check if it's a number referring to search results
                index = int(product_identifier) - 1
                if 0 <= index < len(context.userdata.last_search_results):
                    product = context.userdata.last_search_results[index]
            except ValueError:
                # Try to find by name in last search results
                for p in context.userdata.last_search_results:
                    if product_identifier.lower() in p['name'].lower():
                        product = p
                        break
        
        # If still not found, search all products by name
        if not product:
            for p in products:
                if product_identifier.lower() in p['name'].lower():
                    product = p
                    break
        
        if not product:
            return f"I couldn't find a product matching '{product_identifier}'. Please try browsing our catalog or searching for specific items."
        
        # Check stock
        available_stock = product.get('stock', 0)
        if quantity > available_stock:
            return f"Sorry, we only have {available_stock} units of {product['name']} in stock. Would you like to add {available_stock} to your cart instead?"
        
        # Create order if it doesn't exist
        if context.userdata.current_order is None:
            context.userdata.current_order = Order()
        
        # Add item to order
        context.userdata.current_order.add_item(product, quantity)
        
        total_items = sum(item.quantity for item in context.userdata.current_order.items)
        
        return f"Great! I've added {quantity}x {product['name']} to your cart for ₹{product['price'] * quantity}. Your cart now has {total_items} item{'s' if total_items != 1 else ''} totaling ₹{context.userdata.current_order.total}. Would you like to continue shopping or proceed to checkout?"

    @function_tool
    async def view_current_order(self, context: RunContext[EcommerceData]) -> str:
        """View the current order/cart contents."""
        if not context.userdata.current_order or not context.userdata.current_order.items:
            return "Your cart is currently empty. Would you like to browse our products?"
        
        order = context.userdata.current_order
        result = f"Your current cart ({order.id}):\n\n"
        
        for i, item in enumerate(order.items, 1):
            result += f"{i}. {item.name} - Qty: {item.quantity} - ₹{item.price} each = ₹{item.price * item.quantity}\n"
        
        result += f"\n**Total: ₹{order.total}**\n\n"
        result += "Would you like to add more items, remove something, or proceed to checkout?"
        
        return result

    @function_tool
    async def place_order(self, context: RunContext[EcommerceData]) -> str:
        """Complete the order and save it to file."""
        if not context.userdata.current_order or not context.userdata.current_order.items:
            return "Your cart is empty. Please add some items before placing an order!"
        
        order = context.userdata.current_order
        order.status = "confirmed"
        
        # Add to order history
        context.userdata.orders_history.append(order)
        
        # Save order to file
        try:
            orders_dir = CONFIG["orders_directory"]
            orders_dir.mkdir(exist_ok=True)
            
            filename = f"{order.id}.json"
            filepath = orders_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(order.to_dict(), f, indent=2)
                
            logger.info(f"Order saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving order: {e}")
        
        # Clear current order
        context.userdata.current_order = None
        
        return f"🎉 Order confirmed! Your order {order.id} has been placed successfully.\n\n{order.get_summary()}\n\nThank you for shopping with FreshMart E-Commerce! Your order will be processed and you'll receive updates soon. Is there anything else I can help you with today?"

    @function_tool
    async def view_order_history(self, context: RunContext[EcommerceData]) -> str:
        """View previous orders from this session."""
        if not context.userdata.orders_history:
            return "You haven't placed any orders yet in this session. Would you like to start shopping?"
        
        result = "Your recent orders:\n\n"
        
        for order in context.userdata.orders_history[-3:]:  # Show last 3 orders
            result += f"• {order.get_summary()}\n"
            result += f"  Status: {order.status.title()} | Placed: {order.created_at[:16]}\n\n"
        
        return result + "Would you like to place another order?"

    @function_tool
    async def get_product_details(self, context: RunContext[EcommerceData], product_identifier: str) -> str:
        """Get detailed information about a specific product."""
        # Find the product
        product = None
        products = context.userdata.catalog.get('products', [])
        
        # Try to find by ID first
        for p in products:
            if p['id'] == product_identifier:
                product = p
                break
        
        # If not found by ID, try to find in last search results by index
        if not product and context.userdata.last_search_results:
            try:
                index = int(product_identifier) - 1
                if 0 <= index < len(context.userdata.last_search_results):
                    product = context.userdata.last_search_results[index]
            except ValueError:
                pass
        
        # If still not found, search by name
        if not product:
            for p in products:
                if product_identifier.lower() in p['name'].lower():
                    product = p
                    break
        
        if not product:
            return f"I couldn't find a product matching '{product_identifier}'. Would you like me to search our catalog for you?"
        
        # Build detailed product information
        result = f"**{product['name']}** (ID: {product['id']})\n\n"
        result += f"💰 **Price:** ₹{product['price']}\n"
        result += f"📦 **Stock:** {product.get('stock', 0)} available\n"
        result += f"🏷️ **Category:** {product['category'].title()}\n\n"
        result += f"📝 **Description:** {product['description']}\n\n"
        
        attributes = product.get('attributes', {})
        if attributes:
            result += "🔧 **Features:**\n"
            for attr_name, attr_value in attributes.items():
                result += f"   • {attr_name.replace('_', ' ').title()}: {attr_value}\n"
        
        result += f"\nWould you like to add this {product['name']} to your cart?"
        
        return result


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Initialize e-commerce data
    userdata = EcommerceData()
    
    # Load product catalog
    catalog_file = CONFIG["catalog_file"]
    try:
        if catalog_file.exists():
            with open(catalog_file, 'r', encoding='utf-8') as f:
                userdata.catalog = json.load(f)
        else:
            logger.error(f"Catalog file not found: {catalog_file}")
            userdata.catalog = {'products': []}
    except Exception as e:
        logger.error(f"Error loading catalog: {e}")
        userdata.catalog = {'products': []}
    
    # Create e-commerce agent
    ecommerce_agent = EcommerceAgent()
    
    # Create the session with the e-commerce agent
    session = AgentSession[EcommerceData](
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

    # Start the session with the e-commerce agent
    await session.start(
        agent=ecommerce_agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
